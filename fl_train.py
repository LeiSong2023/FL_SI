import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from config import args

from gcn import BatchGCN
from gat import BatchGAT

import os
import shutil
import logging
from tensorboard_logger import tensorboard_logger

from client import *
from server import *

from data_loader import ChunkSampler
from data_loader import InfluenceDataSet
from data_loader import PatchySanDataSet

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp

args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)





launch_tensorboad = 1
if launch_tensorboad ==1:
    tensorboard_log_dir = 'tensorboard/fl/%s_%s_%s_%s' % (args.file_dir,args.model, args.tensorboard_log,args.vision)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    shutil.rmtree(tensorboard_log_dir)
    tensorboard_logger.configure(tensorboard_log_dir)
    logger.info('tensorboard logging to %s', tensorboard_log_dir)


# adj N*n*n
# feature N*n*f
# labels N*n*c
# Load data
# vertex: vertex id in global network N*n


#get data 
influence_dataset = InfluenceDataSet(args.file_dir, args.dim, args.seed, args.shuffle, args.model)
N = len(influence_dataset)
n_classes = 2
class_weight = influence_dataset.get_class_weight() \
        if args.class_weight_balanced else torch.ones(n_classes)
logger.info("class_weight=%.2f:%.2f", class_weight[0], class_weight[1])

feature_dim = influence_dataset.get_feature_dimension()
n_units = [feature_dim] + [int(x) for x in args.hidden_units.strip().split(",")] + [n_classes]

logger.info("feature dimension=%d", feature_dim)
logger.info("number of classes=%d", n_classes)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


train_start,  valid_start, test_start = \
        0, int(N * args.train_ratio / 100), int(N * (args.train_ratio + args.valid_ratio) / 100)

#train_loader = DataLoader(influence_dataset, batch_size=args.batch,
#                        sampler=ChunkSampler(valid_start - train_start, 0))

train_end = valid_start-1
each_data_size = int(train_end // args.client_num)

valid_loader = DataLoader(influence_dataset, batch_size=args.batch,
                        sampler=ChunkSampler(test_start - valid_start, valid_start))
test_loader = DataLoader(influence_dataset, batch_size=args.batch,
                        sampler=ChunkSampler(N - test_start, test_start))

#create clients
client_list = []
for i in range(args.client_num):
    if args.model == "gcn":
        c_model = BatchGCN(pretrained_emb=influence_dataset.get_embedding(),
                    vertex_feature=influence_dataset.get_vertex_features(),
                    use_vertex_feature=args.use_vertex_feature,
                    n_units=n_units,
                    dropout=args.dropout,
                    instance_normalization=args.instance_normalization)
    elif args.model == "gat":
        n_heads = [int(x) for x in args.heads.strip().split(",")]
        c_model = BatchGAT(pretrained_emb=influence_dataset.get_embedding(),
                vertex_feature=influence_dataset.get_vertex_features(),
                use_vertex_feature=args.use_vertex_feature,
                n_units=n_units, n_heads=n_heads,
                dropout=args.dropout, instance_normalization=args.instance_normalization)
    else:
        raise NotImplementedError
    
    #create dataloader for each client
    train_loader = DataLoader(influence_dataset, batch_size=args.batch,
                        sampler=ChunkSampler(each_data_size, each_data_size*i))
    

    c = Client(client_idx = i, train_loader = train_loader, valid_loader = valid_loader, args = args, device = device, model = c_model)
    client_list.append(c)

if args.cuda:
    class_weight = class_weight.cuda()

#create server
if args.model == "gcn":
    s_model = BatchGCN(pretrained_emb=influence_dataset.get_embedding(),
                vertex_feature=influence_dataset.get_vertex_features(),
                use_vertex_feature=args.use_vertex_feature,
                n_units=n_units,
                dropout=args.dropout,
                instance_normalization=args.instance_normalization)
elif args.model == "gat":
    n_heads = [int(x) for x in args.heads.strip().split(",")]
    s_model = BatchGAT(pretrained_emb=influence_dataset.get_embedding(),
                vertex_feature=influence_dataset.get_vertex_features(),
                use_vertex_feature=args.use_vertex_feature,
                n_units=n_units, n_heads=n_heads,
                dropout=args.dropout, instance_normalization=args.instance_normalization)
else:
    raise NotImplementedError

#server = Server(model = s_model,client_list = client_list,args = args,device = device)
server = Server(model = client_list[0].model,client_list = client_list,args = args,device = device)

for name, param in server.model.named_parameters():
	print(name, '      ', param.size())
# print('----------------')
# for i in range(args.client_num):
#     for name, param in client_list[i].model.named_parameters():
#         print(name, '      ', param.size())
# print(server.model.state_dict()['layer_stack.1.w'])
# print('&&&&&&&&&&&&&&&& this is client para')
# for i,client in enumerate(client_list):
#     print('this is {} client' .format(i))
#     print(client.model.state_dict()['layer_stack.1.w'])
#     print('!!!!!!!!!!!!!!')

# Train model
t_total = time.time()
logger.info("training...")

#federated learning training
for round in range(args.round):
    global_para = server.broadcast_parameters()  #服务器广播当前网络模型
    #clients training
    for index, client in enumerate(client_list):
        client.download_parameters(global_para) #客户端下载全局模型，并更新本地模型
        client.train(class_weight=class_weight,round=round) #客户端训练本地模型
    logger.info("communication round %d client training is finished!", round)  

    #server update
    server.aggregate_para(global_para,client_list)
    server.update_parameters(global_para)

    #server evaluate
    # if (round + 1) % args.check_point == 0:
    best_thr = server.evaluate(dataloader = valid_loader, class_weight=class_weight, round = round, return_best_thr=True, log_desc='valid_')
    server.evaluate(dataloader = test_loader, class_weight = class_weight,round = round, thr=best_thr, log_desc='test_')

logger.info("optimization Finished!")
logger.info("total time elapsed: {:.4f}s".format(time.time() - t_total))

logger.info("retrieve best threshold...")
best_thr = server.evaluate(dataloader = valid_loader, class_weight=class_weight,round = args.round, return_best_thr=True,log_desc='valid_')
# Testing
logger.info("testing...")
server.evaluate(dataloader = test_loader, class_weight=class_weight,round = args.round, thr=best_thr, log_desc='test_')
#server.evaluate(dataloader = test_loader, class_weight=class_weight, thr=-0.5)


