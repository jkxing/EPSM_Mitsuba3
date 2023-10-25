
# from geomloss import SamplesLoss
from sklearn.neighbors import NearestNeighbors
from utils.logger import Logger
import numpy as np
import torch
from geomloss import SamplesLoss

#from experiments.logger import Logger
class Matcher():
    def __init__(self, res, device) -> None:
        self.loss = SamplesLoss("sinkhorn", blur=0.01,scaling=0.9)
        self.device=device
        self.resolution = res
        x = torch.linspace(0, 1, self.resolution)
        y = torch.linspace(0, 1, self.resolution)
        pos = torch.meshgrid(x, y)
        self.pos = torch.cat([pos[1][..., None], pos[0][..., None]], dim=2).to(self.device).reshape(-1,2)
        self.pos_grad = torch.cat([pos[1][..., None], pos[0][..., None]], dim=2).to(self.device).reshape(-1,2).requires_grad_()
        self.pos_np = self.pos.clone().cpu().numpy().reshape(-1,2)
        
        self.num_vectors = 50
        
        self.num_principle_vectors = 3
        self.rgb_weight = 1

    def visualize_point_sink(self, res, match):#(N,5) (r,g,b,x,y)
        res = res.detach(). cpu().numpy()
        match = match.detach().cpu().numpy()
        res_2d = res.reshape(self.resolution,self.resolution,5)
        #np.savetxt(f"X.txt",res[...,3],fmt="%.1f")
        #np.savetxt(f"Y.txt",res.reshape(self.resolution,self.resolution,5)[...,4],fmt="%.1f")
        imgx = np.zeros((self.resolution,self.resolution,3))
        imgx[...,0] = (res_2d[...,3]>0)*res_2d[...,3]
        imgx[...,1] = (res_2d[...,3]<0)*res_2d[...,3]*-1
        Logger.save_img("imgx.png", imgx, flip=False)
        imgx[...,0] = (res_2d[...,4]>0)*res_2d[...,4]
        imgx[...,1] = (res_2d[...,4]<0)*res_2d[...,4]*-1
        Logger.save_img("imgy.png", imgx, flip=False)
        X = match[...,3:]
        #need install sklearn
        nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(self.pos_np)
        distances = np.exp(-distances*self.resolution)
        img = np.sum(match[indices,:3]*distances[...,None],axis = 1)
        img = img/np.sum(distances,axis = 1)[...,None]
        img = img.reshape(self.resolution, self.resolution, 3)
        Logger.save_img("visualizematch.png",img, flip=False)
        #exit()
    
    def match_Sinkhorn(self, render_point, gt_rgb):
        target_point_5d = torch.zeros((gt_rgb.shape[0], 5), device=self.device)
        target_point_5d[..., :3] = torch.clamp(gt_rgb,0,1)
        target_point_5d[..., 3:] = self.pos.clone().detach()
        render_point_5d = torch.zeros((render_point.shape[0], 5), device=self.device)
        render_point_5d[..., :3] = torch.clamp(render_point,0,1)
        render_point_5d[..., 3:] = self.pos.clone().detach()
        render_point_5d.requires_grad_(True)
        pointloss = self.loss(render_point_5d, target_point_5d)
        [g] = torch.autograd.grad(torch.sum(pointloss)*self.resolution*self.resolution, [render_point_5d])
        #match = (render_point_5d-g).detach()
        #self.visualize_point_sink(-g, match)
        return g

    def visualize_point(self, res, title, view):#(N,5) (r,g,b,x,y)
        res = res.detach().cpu().numpy()
        #need install sklearn
        # nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(X)
        # distances, indices = nbrs.kneighbors(self.pos_np)
        # distances = np.exp(-distances*self.resolution)
        # img = np.sum(res[indices,:3]*distances[...,None],axis = 1)
        # img = img/np.sum(distances,axis = 1)[...,None]
        # img = img.reshape(self.resolution, self.resolution, 3)
        self.logger.add_image(title+"_"+str(view), res[:, :3].reshape(self.resolution, self.resolution, 3), self.step, flip=False)

    def match_sliced_wasserstein(self, render_point, gt_rgb):
        target_point_5d = torch.zeros((gt_rgb.shape[0], 5), device=self.device)
        target_point_5d[..., :3] = torch.clamp(gt_rgb,0,1) * self.rgb_weight
        target_point_5d[..., 3:] = self.pos.clone().detach()
        render_point_5d = torch.zeros((render_point.shape[0], 5), device=self.device)
        render_point_5d[..., :3] = torch.clamp(render_point,0,1).clone().detach() * self.rgb_weight
        render_point_5d[..., 3:] = self.pos.clone().detach()
        render_point_5d.requires_grad_(True)
        #render_point_5d_match = render_point_5d.clone().reshape(-1,h*w,5)
        #render_point_5d_match.clamp_(0.0,1.0)

        target_point_5d_clone = target_point_5d.clone().detach()
        if self.num_principle_vectors > 0:
            assert self.num_principle_vectors <= 3
            #print(self.num_principle_vectors)
            U, S, V = torch.pca_lowrank(target_point_5d[:, :3], q=3)
            target_point_principle = torch.cat((torch.matmul(target_point_5d[:, :3], V[:, :self.num_principle_vectors]), target_point_5d[:, 3:]), dim=1)
            render_point_principle = torch.cat((torch.matmul(render_point_5d[:, :3], V[:, :self.num_principle_vectors]), render_point_5d[:, 3:]), dim=1)
            #print(V)
        else:
            target_point_principle = target_point_5d
            render_point_principle = render_point_5d

        len_vector = 2 + (3 if self.num_principle_vectors == 0 else self.num_principle_vectors)

        V = torch.rand((len_vector, self.num_vectors), device=render_point_principle.device) * 2.0 - 1.0
        V = torch.nn.functional.normalize(V, p=2, dim=0)

        projected_render = torch.matmul(render_point_principle, V).squeeze(0) # h*w, num_vectors
        projected_gt = torch.matmul(target_point_principle, V).squeeze(0)

        sorted_render, render_indices = torch.sort(projected_render, dim=0, stable=True)
        sorted_gt, gt_indices = torch.sort(projected_gt, dim=0, stable=True)

        # pointloss = self.loss(render_point_5d, target_point_5d)
        pointloss = torch.sum((sorted_render - sorted_gt) ** 2)

        [g] = torch.autograd.grad(pointloss, [render_point_5d])
        g[:, :3] /= self.rgb_weight

        # # print(g.min(), g.max())

        # # nlogn implementation of permutation inverse, can be optimized
        # render_rank = torch.argsort(render_indices, dim=0)
        # matched_indices = torch.zeros((self.num_vectors, render_rank.shape[0]), dtype=torch.long, device=render_indices.device)  # num_vectors, h*w
        # for i in range(self.num_vectors):
        #     matched_indices[i, :] = gt_indices[render_rank[:, i], i]
        #     # print(matched_indices[i, :].min(), matched_indices[i, :].max(), matched_indices[i, :].float().mean())

        # # render_point_5d_match = render_point_5d_match.squeeze(0)  # h*w, 5
        # target_point_5d_clone = target_point_5d_clone.squeeze(0) # h*w, 5
        # # print(target_point_5d.shape)
        # # # return torch.mean((render_point_5d_match[None, :] - target_point_5d[matched_indices, :]) ** 2)
        # # return torch.mean((render_point_5d_match - torch.mean(target_point_5d[matched_indices, :], dim=0)) ** 2)

        # match_point_5d = target_point_5d_clone[matched_indices, :]
        # match_point_5d_mean = torch.mean(match_point_5d, dim=0)
        # # print(matched_indices)
        # # print(match_point_5d.shape)

        # mean_rgb = match_point_5d_mean[..., :3]
        # mean_xy = match_point_5d_mean[..., 3:]

        # #print(mean_rgb.max(), mean_rgb.min())
        # self.save_image(mean_rgb / self.rgb_weight, title=f"mean_rgb_{self.num_principle_vectors}", view=0)
        # self.save_image(g / self.rgb_weight, title=f"gradient_rgb", view=0)

        # mean_xy = mean_xy.clone().detach().cpu().numpy()
        # mean_x = np.concatenate((mean_xy[:, 0:1], np.zeros((mean_xy.shape[0], 2))), axis=1)
        # mean_y = np.concatenate((mean_xy[:, 1:2], np.zeros((mean_xy.shape[0], 2))), axis=1)

        # Logger.add_image("mean_x", mean_x.reshape((self.resolution, self.resolution, 3)), step=0, flip=False)
        # Logger.add_image("mean_y", mean_y.reshape((self.resolution, self.resolution, 3)), step=0, flip=False)

        # import matplotlib.pyplot as plt
        # import io
        # render_rgb_copy = render_point_5d[:, :3].clone().detach().cpu().numpy().reshape((self.resolution, self.resolution, 3))
        # fig, ax = plt.subplots()
        # ax.imshow(render_rgb_copy)

        # mean_x = mean_x.reshape((self.resolution, self.resolution, 3))
        # mean_y = mean_y.reshape((self.resolution, self.resolution, 3))

        # cmap = plt.cm.get_cmap("hsv", len(range(0, self.resolution, 15)) ** 2)
        # t = 0
        # # for i in range(0, self.resolution, 15):
        # #     for j in range(0, self.resolution, 15):
        # #         #print(i, j, mean_x[i, j, 0] * self.resolution, mean_y[i, j, 0] * self.resolution)
        # #         ax.arrow(i, j, mean_x[i, j, 0] * self.resolution - i, mean_y[i, j, 0] * self.resolution - j, color=cmap(t))
        # #         t = t + 1
        
        # # io_buf = io.BytesIO()
        # # fig.savefig(io_buf, format='raw')
        # # io_buf.seek(0)
        # # img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
        # #                     newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        # # io_buf.close()
        # # img_arr = img_arr[:, :, :3] # throw away A channel
        # # Logger.add_image("match_plot", img_arr, flip=False)

        # fig.savefig("match_plot.png")

        # plt.close(fig)

        return g
    
    def visualize_point_sink_exp(self, res):#(N,5) (r,g,b,x,y)
        res = res.detach().cpu().numpy()
        res_2d = res.reshape(self.resolution,self.resolution,2)
        imgx = np.zeros((self.resolution,self.resolution,3))
        imgx[...,0] = (res_2d[...,0]>0)*res_2d[...,0]
        imgx[...,1] = (res_2d[...,0]<0)*res_2d[...,0]*-1
        Logger.save_img("imgx.png", imgx, flip=False)
        imgx[...,0] = (res_2d[...,1]>0)*res_2d[...,1]
        imgx[...,1] = (res_2d[...,1]<0)*res_2d[...,1]*-1
        Logger.save_img("imgy.png", imgx, flip=False)
        #exit()

    def match_Sinkhorn_exp(self, render_img, gt_rgb, t=1):
        #print(torch.max(render_point),torch.min(render_point),gt_rgb)
        #exit()
        #render_point = torch.zeros((render_point.shape[0], 3), device=self.device)
        #render_point[..., :2] = self.pos.clone().detach()
        match_res = []
        for i in range(3):
            target_weight = gt_rgb[...,i]
            source_weight = render_img[...,i]
            cost = self.loss(source_weight, self.pos_grad, target_weight, self.pos)
            [g_i] = torch.autograd.grad(cost, [self.pos_grad])
            g_i = g_i/(source_weight.reshape(-1,1)+0.0001)
            #g_i_ = g_i.reshape(self.resolution,self.resolution,-1)
            #g_i_ = torch.repeat_interleave(g_i_, t, dim=0)
            #g_i_ = torch.repeat_interleave(g_i_, t, dim=1)
            match_res.append(g_i)
            #self.visualize_point(-g_i, i, render_img)
        #print(render_img.shape,match_res[0].shape)
        grad_i = (render_img[...,0:1]*match_res[0]+render_img[...,1:2]*match_res[1]+render_img[...,2:]*match_res[2])/(torch.sum(render_img,dim=-1,keepdim=True)+0.0001)
        print(grad_i)
        self.visualize_point_sink_exp(-grad_i)
        print("sdlfjds")
        exit()
        grad_i = torch.cat([render_img,grad_i],dim=-1)
        return grad_i
    def save_image(self, res, title, view):#(N,5) (r,g,b,x,y)
        res = res.detach().cpu().numpy()
        Logger.add_image(title+"_"+str(view), res[:, :3].reshape(self.resolution, self.resolution, 3), step=self.num_principle_vectors, flip=False)

