# cd ~/dev/

cd systems
source fbn2_chain.tcl
cd ..

mol delrep 4 0
mol delrep 3 0
mol delrep 0 0


mol modstyle 0 0 Licorice 10.01 12.0 50.0

mol color ColorID 8
mol material Diffuse
mol representation Licorice 10.0 12.0 50.0
mol selection "name NUC"
mol addrep top

display resize 2000 2000
display cuedensity 0.300000
animate goto 16

exit # FLIP THE COLORMAP!

# (0, 5000)
mol modselect 0 0 name NUC and index 0 to 5000
mol modselect 1 0 name NUC and index 0 to 5000
mol scaleminmax top 0 0.000000 10.000000 
mol scaleminmax top 1 0.000000 10.000000

molinfo top set rotate_matrix {{{-0.0704775 -0.997503 -0.00369523 0} {-0.788998 0.0580124 -0.611645 0} {0.610334 -0.0401916 -0.791118 0} {0 0 0 1}}}
molinfo top set center {{88.043198 39.018162 25.256498}}
molinfo top set global_matrix {{{1 0 0 0.12} {0 1 0 -0.16} {0 0 1 0} {0 0 0 1}}}
molinfo top set scale_matrix {{{0.000225396 0 0 0} {0 0.000225396 0 0} {0 0 0.000225396 0} {0 0 0 1}}}

render TachyonInternal hierarchy_fbn2_0.tga

# (200, 2700)
# chain colored 0 to 10, so want colorscale to be 10*start/5000 to 10*end/5000
mol modselect 0 0 name NUC and index 200 to 2700
mol modselect 1 0 name NUC and index 200 to 2700
mol scaleminmax top 0 0.400000 5.400000 
mol scaleminmax top 1 0.400000 5.400000 

molinfo top set rotate_matrix {{{-0.0704775 -0.997503 -0.00369523 0} {-0.788998 0.0580124 -0.611645 0} {0.610334 -0.0401916 -0.791118 0} {0 0 0 1}}}
molinfo top set center {{88.043198 39.018162 25.256498}}
molinfo top set global_matrix {{{1 0 0 -0.66} {0 1 0 -0.06} {0 0 1 0} {0 0 0 1}}}
molinfo top set scale_matrix {{{0.00046957 0 0 0} {0 0.00046957 0 0} {0 0 0.00046957 0} {0 0 0 1}}}

render TachyonInternal hierarchy_fbn2_1.tga

# (500, 1500)
mol modselect 0 0 name NUC and index 500 to 1500
mol modselect 1 0 name NUC and index 500 to 1500
mol scaleminmax top 0 1.000000 3.000000 
mol scaleminmax top 1 1.000000 3.000000 

molinfo top set rotate_matrix {{{-0.0704775 -0.997503 -0.00369523 0} {-0.788998 0.0580124 -0.611645 0} {0.610334 -0.0401916 -0.791118 0} {0 0 0 1}}}
molinfo top set center {{88.043198 39.018162 25.256498}}
molinfo top set global_matrix {{{1 0 0 -0.7} {0 1 0 0.24} {0 0 1 0} {0 0 0 1}}}
molinfo top set scale_matrix {{{0.000583435 0 0 0} {0 0.000583435 0 0} {0 0 0.000583435 0} {0 0 0 1}}}

render TachyonInternal hierarchy_fbn2_2.tga

# (750, 1000)
mol modselect 0 0 name NUC and index 750 to 1000
mol modselect 1 0 name NUC and index 750 to 1000
mol scaleminmax top 0 1.500000 2.000000 
mol scaleminmax top 1 1.500000 2.000000 

molinfo top set rotate_matrix {{{-0.0704775 -0.997503 -0.00369523 0} {-0.788998 0.0580124 -0.611645 0} {0.610334 -0.0401916 -0.791118 0} {0 0 0 1}}}
molinfo top set center {{88.043198 39.018162 25.256498}}
molinfo top set global_matrix {{{1 0 0 -0.7} {0 1 0 0.24} {0 0 1 0} {0 0 0 1}}}
molinfo top set scale_matrix {{{0.000583435 0 0 0} {0 0.000583435 0 0} {0 0 0.000583435 0} {0 0 0 1}}}

render TachyonInternal hierarchy_fbn2_3.tga

