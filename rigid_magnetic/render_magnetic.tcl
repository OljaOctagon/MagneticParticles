#!/usr/bin/env tclsh
# ============================================================
#  VMD visualization script for LAMMPS trajectory
#  Usage: vmd -e visualize_lammps.tcl -args <trajectory.lammpstrj>
# ============================================================

# --- Load trajectory from command-line argument ---------------
set traj [lindex $argv 0]
if {$traj eq ""} {
    puts "Usage: vmd -e visualize_lammps.tcl -args <trajectory.lammpstrj>"
    quit
}

mol new $traj type lammpstrj waitfor all

# --- Display settings -----------------------------------------
display backgroundgradient off
color Display Background white
display depthcue                off
display culling                 off
display projection              Orthographic
display rendermode              Acrobat3D
display axis off
axes location                   Off


# --- Representation: Type 1  →  VDW, scale 0.3, pink (ColorID 9) ---
mol delrep 0 top

mol representation VDW 0.3 32.0
mol color ColorID 9
mol selection {type 1}
mol material AOChalky
mol addrep top

# --- Representation: Type 2  →  VDW, scale 0.2, silver (ColorID 6) ---
mol representation VDW 0.2 32.0
mol color ColorID 6
mol selection {type 2}
mol material AOChalky
mol addrep top

# --- Fit all atoms into view ----------------------------------
display resetview
mol drawframes top 0 {now}
scale to 1.0
rotate x by -90
display update

# --- Frame the system in X and Y (reset view, then fit) -------
# Center and zoom so all atoms are visible
set sel [atomselect top "all"]
set minmax [measure minmax $sel]
set min [lindex $minmax 0]
set max [lindex $minmax 1]

# Compute centre
set cx [expr {([lindex $max 0] + [lindex $min 0]) / 2.0}]
set cy [expr {([lindex $max 1] + [lindex $min 1]) / 2.0}]
set cz [expr {([lindex $max 2] + [lindex $min 2]) / 2.0}]

# Translate so the system is centred in the viewport
molinfo top set center_matrix \
    [list [transoffset [list [expr {-$cx}] [expr {-$cy}] [expr {-$cz}]]]]

# Scale to fill the view (X/Y extent)
set dx [expr {[lindex $max 0] - [lindex $min 0]}]
set dy [expr {[lindex $max 1] - [lindex $min 1]}]
set extent [expr {$dx > $dy ? $dx : $dy}]
if {$extent > 0} {
    set s [expr {1.8 / $extent}]          ;# 1.8 leaves a small border
    scale to $s
}

$sel delete

display update

# --- Render with Tachyon Internal -----------------------------
# High-resolution render (adjust filename/size as needed)

render POV3 render_outputxz.pov

puts "Done. Output written to render_output.pov"
exit