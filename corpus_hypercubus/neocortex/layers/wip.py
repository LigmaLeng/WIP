#####
#WIP#
#####
# def dilate2D(self, dv):
#     idr = xp.identity(self.out_h, dtype=bool)
#     idc = xp.identity(self.out_w, dtype=bool)
#     for i in range(self.out_h - 1, 0, -1):
#         if self.row_stride > 1:
#             idr = np.insert(idr, i, np.zeros((rs - 1, self.out_h), dtype=bool), axis=0)
      
#     for i in range(self.out_w - 1, 0, -1):
#             if self.col_stride > 1:
#             idc = np.insert(idc, i, np.zeros((cs - 1, self.out_w), dtype=bool), axis=1)

#     return np.pad(idr @ dv @ idc, pad_width=((0,0), (0,0), (1,0), (1,0)), mode="constant", constant_values=(0,0))

# def dilate1D(self, dv):
#     if self.v_k > 1 and self.row_stride > 1:
#         for i in range(self.v_k - 1, 0, -1):
#             dv = xp.insert(dv, i, xp.zeros((self.row_stride - 1, self.v_l), dtype=bool), axis=-2)

#     elif self.v_l > 1 and self.col_stride > 1:
#         for i in range(self.v_l - 1, 0, -1):
#             dv = xp.insert(dv, i, xp.zeros((self.v_k, self.col_stride - 1), dtype=bool), axis=-1)
#     return dv