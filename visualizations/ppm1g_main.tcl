cd systems
source ppm1g_dbscan.tcl
source ppm1g_ins.tcl
cd ..

mol showrep 0 0 0
mol showrep 1 0 0

mol modstyle 3 0 Licorice 10.000001 12.000000 50.000000
mol modstyle 4 0 VDW 1.00001 50.0

mol modstyle 3 1 Licorice 10.000001 12.000000 50.000000
mol modstyle 4 1 VDW 1.00001 50.0

display resize 2000 2000
display cuedensity 0.300000

animate goto 16

# Zoomed out view
molinfo top set rotate_matrix {{{0.976038 -0.118235 0.18266 0} {0.182997 0.900206 -0.395138 0} {-0.117713 0.419097 0.900274 0} {0 0 0 1}}}
molinfo top set center {{88.043198 39.018162 25.256498}}
molinfo top set global_matrix {{{1 0 0 -0.24} {0 1 0 -4.47035e-08} {0 0 1 0} {0 0 0 1}}}
molinfo top set scale_matrix {{{0.000374887 0 0 0} {0 0.000374887 0 0} {0 0 0.000374887 0} {0 0 0 1}}}

render TachyonInternal ppm1g_zoomout_dbscan.tga

molinfo top set rotate_matrix {{{0.976038 -0.118235 0.18266 0} {0.182997 0.900206 -0.395138 0} {-0.117713 0.419097 0.900274 0} {0 0 0 1}}}
molinfo top set center {{88.043198 39.018162 25.256498}}
molinfo top set global_matrix {{{1 0 0 -3.44} {0 1 0 -0.6} {0 0 1 0} {0 0 0 1}}}
molinfo top set scale_matrix {{{0.00111941 0 0 0} {0 0.00111941 0 0} {0 0 0.00111941 0} {0 0 0 1}}}

render TachyonInternal ppm1g_top_dbscan.tga

# Whole small trace

mol modselect 1 1 "name NUC and index 1000 to 1500"
mol modselect 2 1 "name NUC and index 1000 to 1500"
mol modselect 3 1 "user < 0 and index 1000 to 1500"
mol modselect 4 1 "user < 0 and index 1000 to 1500"

mol color ColorID 2
mol material Transparent
mol representation Licorice 10.0 12.0 50.0
mol selection "name NUC"
mol addrep top

molinfo top set rotate_matrix {{{0.976038 -0.118235 0.18266 0} {0.182997 0.900206 -0.395138 0} {-0.117713 0.419097 0.900274 0} {0 0 0 1}}}
molinfo top set center {{88.043198 39.018162 25.256498}}
molinfo top set global_matrix {{{1 0 0 -1.84} {0 1 0 0.22} {0 0 1 0} {0 0 0 1}}}
molinfo top set scale_matrix {{{0.000674755 0 0 0} {0 0.000674755 0 0} {0 0 0.000674755 0} {0 0 0 1}}}

render TachyonInternal ppm1g_small_trace.tga