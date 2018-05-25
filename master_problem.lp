\ Model cutstock
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  x[0] + x[1] + x[2] + x[3] + x[4] + C5 + C6 + C7 + C8 + C9 + C10 + C11
Subject To
 c1[0]: 2 x[0] + 2 C5 + C10 + C11 >= 12
 c1[1]: 2 x[1] + 2 C6 + C9 + C10 >= 6
 c1[2]: 7 x[2] + C5 + 7 C7 + 6 C8 + 2 C9 + 4 C11 >= 16
 c1[3]: 11 x[3] + C7 + 2 C10 >= 1
 c1[4]: 10 x[4] + C5 + C6 + 2 C8 + 3 C9 + C11 >= 17
Bounds
Generals
 x[0] x[1] x[2] x[3] x[4] C5 C6 C7 C8 C9 C10 C11
End
