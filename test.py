import torch
from torchvision import datasets, transforms
import time
from data import load_data
from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler


if __name__ == '__main__':
    # data, dataset = load_data('products')

    # cluster_data = ClusterData(data, num_parts=15000, recursive=False, save_dir=dataset.processed_dir)
    # for num_workers in range(2,10,2): 
    #     dataloader = ClusterLoader(cluster_data, batch_size=128,
    #                             shuffle=True, num_workers=num_workers)
    #     subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1],
    #                                     batch_size=1024, shuffle=False,
    #                                     num_workers=num_workers)
    #     start = time.time()
    #     for epoch in range(1, 5):
    #         for data in dataloader:
    #             pass

    #     end = time.time()
    #     print("Finish with:{} second, num_workers={}".format(end-start,num_workers))

    use_cuda = torch.cuda.is_available()

    for num_workers in range(0,100,5):  # 遍历worker数
        kwargs = {'num_workers': num_workers, 'pin_memory': False} if use_cuda else {}
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=64, shuffle=True, **kwargs)



        start = time.time()
        for epoch in range(1, 5):
            for batch_idx, (data, target) in enumerate(train_loader): # 不断load
                pass
        end = time.time()
        print("Finish with:{} second, num_workers={}".format(end-start,num_workers))