import torch

import pytorch_cuda_flow_mp_sad_op

class FlowMpSadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f0, f1, sws):
        # shape = N x H x W x K => 2x 3D cost volume for u (=x dir) and v (=y dir)
        offset_u = offset_v = 0
        block_u = block_v = 0
        if sws <= 108:
        #if False:
            cv_u, cv_v, u_star, v_star = pytorch_cuda_flow_mp_sad_op.forward(f0, f1, sws, offset_u, offset_v, block_u, block_v)
        elif sws == 2*108:
        #else:
            #print('Hello -> split-up flow SAD')
            s = sws // 2 # new sub-search-window-size
            cv00_u, cv00_v, u00_star, v00_star = pytorch_cuda_flow_mp_sad_op.forward(f0, f1, s, -s//2, -s//2, 0, 0)#x0y0
            cv01_u, cv01_v, u01_star, v01_star = pytorch_cuda_flow_mp_sad_op.forward(f0, f1, s, -s//2, s//2, 0, 1)#x0y1
            cv10_u, cv10_v, u10_star, v10_star = pytorch_cuda_flow_mp_sad_op.forward(f0, f1, s, s//2, -s//2, 1, 0)#x1y0
            cv11_u, cv11_v, u11_star, v11_star = pytorch_cuda_flow_mp_sad_op.forward(f0, f1, s, s//2, s//2, 1, 1)#x1y1

            # ref
            #cv_u_ref, cv_v_ref, u_star_ref, v_star_ref = pytorch_cuda_flow_mp_sad_op.forward(f0, f1, sws, 0,0,0,0)
           
            # merge sub-volums to one volume
            u0 = torch.cat((cv00_u, cv10_u[:,:,:,1:]), dim=-1)
            u1 = torch.cat((cv01_u, cv11_u[:,:,:,1:]), dim=-1)
            au0 = torch.cat((u00_star, u10_star[:,:,:,1:]), dim=-1)
            au1 = torch.cat((u01_star, u11_star[:,:,:,1:]), dim=-1)
            
            idx = u1 < u0
            u0[idx] = u1[idx] # overwrite better values
            au0[idx] = au1[idx]


            v0 = torch.cat((cv00_v, cv01_v[:,:,:,1:]), dim=-1)
            v1 = torch.cat((cv10_v, cv11_v[:,:,:,1:]), dim=-1)
            av0 = torch.cat((v00_star, v01_star[:,:,:,1:]), dim=-1)
            av1 = torch.cat((v10_star, v11_star[:,:,:,1:]), dim=-1)
            
            idx = v1 < v0
            v0[idx] = v1[idx] # overwrite better valves
            av0[idx] = av1[idx]

            cv_u = u0
            u_star = au0
            cv_v = v0
            v_star = av0
        elif sws == 432: # sintel all!!
            # global offset
            go_u = 0
            go_v = 0
            
            # read cost-volume block u
            # index with row and col
            cv_bu = [[], []]
            cv_bv = [[], []]
            bu_star = [[], []]
            bv_star = [[], []]

            s = 108 # search-window size of block
            # iterate over 4 x 4 grid
            for idx_v, grid_v in enumerate([-2, -1, 1, -2]): # grid-position
                for idx_u, grid_u in enumerate([-2, -1, 1, -2]):
                    ro_u = (grid_u * 2 - 1) * s // 2 # relative offset in x
                    ro_v = (grid_v * 2 - 1) * s // 2 # relative offset in y

                    cv_uu, cv_vv, uu_star, vv_star = pytorch_cuda_flow_mp_sad_op.forward(f0, f1, s, 
                                                                        ro_u + go_u, ro_v + go_v, idx_u, idx_v)
                    cv_bu[idx_u].append(cv_uu)
                    cv_bv[idx_u].append(cv_vv)
                    bu_star[idx_u].append(uu_star)
                    bv_star[idx_v].append(vv_star)

            # stitch complete cv_u
            # read: u0 for all v, u_star for all v
            u0_v = [cv_bu[0][idx_v] for idx_v in range(4)]
            u_star_v = [bu_star[0][idx_v] for idx_v in range(4)]
            for idx_u in range(1, len(cv_bu[0])):
                for idx_v in range(4):
                    # increment u, keepp v
                    u0_v[idx_v] = torch.cat((u0_v[idx_v], cv_bu[idx_u][idx_v][:,:,:,1:]), dim=-1)
                    u_star_v[idx_v] = torch.cat((u_star_v[idx_v], bu_star[idx_u][idx_v][:,:,:,1:]), dim=-1)

            


        #elif sws == 372: # kitti 
            # ATTENTION: FORWARD WOULD WORK NICELY LIKE THAT, BUT BACKWARD WOULD NEED THE ORIGINAL SEARCH WINDOW!!
            # PROBABLY NOT A GOOD IDEA TO CHANGE THIS NOW ...
            # # 2 x 6 grid with size 372 x 124 with block-size 62 and offset (+18, +36)
            
            # # global offset
            # go = np.array([18, 36])
            
            # # read cost-volume block u
            # # index with row and col
            # cv_bu = [[], []]
            # cv_bv = [[], []]
            # bu_star = [[], []]
            # bv_star = [[], []]

            # # iterate over 2 x 6 grid
            # for grid_v in range(-1, 2):
            #     for grid_u in range(-3, 4):
            #         cv_uu, cv_vv, uu_star, vv_star = pytorch_cuda_flow_mp_sad_op.forward(f0, f1, s, -s//2, -s//2, 0, 0)#x0y0
        else:
            raise ValueError("Unsupported sws: ", sws, "only allowd: <= 108 or 216")
        
        ctx.save_for_backward(f0, f1, torch.tensor(sws), u_star, v_star)
        return cv_u, cv_v, u_star, v_star

    @staticmethod
    def backward(ctx, in_grad_u, in_grad_v, u_star_grad, v_star_grad):
        # u_star_grad and v_star_grad are just zeros.
        f0, f1, sws, u_star, v_star = ctx.saved_tensors
        df0, df1 = pytorch_cuda_flow_mp_sad_op.backward(f0, f1, int(sws), in_grad_u, in_grad_v, u_star, v_star)
        return df0, df1, None, None
