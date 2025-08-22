mol new ../../../share/mei_runs/ppm1g/v2/010_topology.0.psf waitfor all
mol addfile ../../../share/mei_runs/ppm1g/v2/010_trajectory.0.dcd step 1100 first 0 last -1  waitfor all
mol addfile ../../../share/mei_runs/ppm1g/v2/010_trajectory.1.dcd step 1100 first 0 last -1  waitfor all
mol addfile ../../../share/mei_runs/ppm1g/v2/010_trajectory.2.dcd step 1100 first 0 last -1  waitfor all
mol addfile ../../../share/mei_runs/ppm1g/v2/010_trajectory.3.dcd step 1100 first 0 last -1  waitfor all
set sel [atomselect top "name NUC"]
set nf [molinfo top get numframes]
set fp [open ppm1g_dbscan_colors.dat r]
set line ""
        for {set i 0} {$i < $nf} {incr i} {
          gets $fp line
          $sel frame $i
          $sel set user $line
        }
        close $fp
        $sel delete

        mol color User
        mol material Diffuse
        mol representation Licorice 10.0 12.0 50.0
        mol selection "name NUC"
        mol addrep top
        set numReps [molinfo top get numreps]
        set lastRep [expr {$numReps - 1}]
        mol scaleminmax top $lastRep 0.000000 10.000000
        mol colupdate $lastRep top  on

        mol color User
        mol material Diffuse
        mol representation vdw 1.0 50.0
        mol selection "name NUC"
        mol addrep top
        set numReps [molinfo top get numreps]
        set lastRep [expr {$numReps - 1}]
        mol scaleminmax top $lastRep 0.000000 10.000000
        mol colupdate $lastRep top on

        mol color ColorID 2
        mol material Diffuse
        mol representation Licorice 10.0 12.0 50.0
        mol selection "user < 0"
        mol addrep top
        set numReps [molinfo top get numreps]
        set lastRep [expr {$numReps - 1}]
        mol colupdate $lastRep top on
        mol selupdate $lastRep top on

        mol color ColorID 2
        mol material Diffuse
        mol representation vdw 1.0 50.0
        mol selection "user < 0"
        mol addrep top
        set numReps [molinfo top get numreps]
        set lastRep [expr {$numReps - 1}]
        mol colupdate $lastRep top on
        mol selupdate $lastRep top on

        color scale method Turbo
        set sel [atomselect top "name NUC CAP"]
$sel set radius 55.0
set sel [atomselect top "name DNA"]
$sel set radius 25.0
