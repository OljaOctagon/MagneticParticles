# load_lammpstrj_type1_user_only.tcl
# Usage:
#   vmd -e load_lammpstrj_type1_user_only.tcl -args traj.lammpstrj cluster_type_user.dat
#
# cluster_type_user.dat: T lines x Ntype1 columns (ONLY atoms with type 1)

proc load_user_table_into_type1 {molid tablefile} {
    set fp [open $tablefile r]

    set nframes [molinfo $molid get numframes]
    set sel0 [atomselect $molid "type 1" frame 0]
    set ntype1 [$sel0 num]
    $sel0 delete

    puts "Molecule: $molid, nframes=$nframes, n(type 1)=$ntype1"
    puts "Reading user table for type 1 only: $tablefile"

    set t 0
    while {[gets $fp line] >= 0} {
        if {$t >= $nframes} {
            puts "Warning: table has more lines than frames; stopping at frame $t"
            break
        }

        set vals [split [string trim $line]]

        if {[llength $vals] == 0} {
            puts "Warning: empty line at t=$t; skipping"
            incr t
            continue
        }

        if {[llength $vals] != $ntype1} {
            puts "ERROR: at line/frame $t, expected $ntype1 values but got [llength $vals]"
            puts "Your table must be T lines x N(type 1) columns."
            close $fp
            return
        }

        animate goto $t
        set sel [atomselect $molid "type 1" frame $t]
        # assigns values in the *order of the selection* (stable if dump sorted by id)
        $sel set user $vals
        $sel delete

        incr t
    }

    close $fp
    puts "Loaded user values for $t frame(s) onto type 1 atoms."
}

# ---------------- Main ----------------

if {$argc < 2} {
    puts "Usage: vmd -e load_lammpstrj_type1_user_only.tcl -args traj.lammpstrj cluster_type_user.dat"
    exit
}

set dumpfile  [lindex $argv 0]
set tablefile [lindex $argv 1]

# Load LAMMPS dump trajectory
mol new $dumpfile type lammpstrj waitfor all
set molid [molinfo top]

# Load user field onto type 1 atoms only
load_user_table_into_type1 $molid $tablefile

# -------- Discrete hex colors for integer User codes via multiple reps --------

proc hex2rgb {hex} {
    # Clean up common issues: quotes, whitespace
    set hex [string trim $hex "\"' \t\r\n"]

    # Allow passing without leading '#'
    if {[string index $hex 0] ne "#"} {
        set hex "#$hex"
    }

    # Basic validation
    if {[string length $hex] != 7} {
        error "hex2rgb: expected #RRGGBB, got '$hex'"
    }

    # Parse; scan returns number of successful conversions
    if {[scan [string range $hex 1 2] %x r] != 1} { error "hex2rgb: bad R in '$hex'" }
    if {[scan [string range $hex 3 4] %x g] != 1} { error "hex2rgb: bad G in '$hex'" }
    if {[scan [string range $hex 5 6] %x b] != 1} { error "hex2rgb: bad B in '$hex'" }

    return [list [expr {$r/255.0}] [expr {$g/255.0}] [expr {$b/255.0}]]
}


# Your palette: code -> hex
set user_colors {
    0 #86878A
    1 #591FDF
    2 #000000
    3 #EEBF25
    4 #1CA879
    5 #ff7f00
    6 #a65628
    7 #da399c
}

# Define ColorID 0..7 to your hex values (safe; we only need 8)
foreach {code hex} $user_colors {
    set rgb [hex2rgb $hex]
    color change rgb $code {*}$rgb
}

set molid [molinfo top]

# Remove any default rep(s)
set nrep [molinfo $molid get numreps]
for {set r [expr {$nrep-1}]} {$r >= 0} {incr r -1} {
    mol delrep $r $molid
}

# Create one rep per class (discrete)
# NOTE: use a tolerance so floats like 1.000 still match robustly
for {set code 0} {$code <= 7} {incr code} {
    mol addrep $molid
    set rep [expr {[molinfo $molid get numreps] - 1}]

    set lo [expr {$code - 0.5}]
    set hi [expr {$code + 0.5}]

    mol modselect $rep $molid "type 1 and user > $lo and user <= $hi"
    mol modstyle  $rep $molid "VDW 0.4 30.0"
    mol modcolor  $rep $molid "ColorID $code"
    mol modmaterial $rep $molid Opaque
}

puts "Discrete coloring: one rep per User code (0..7), hex palette applied."

# -------- View & display settings --------

# Orthographic projection (no perspective distortion)
display projection Orthographic

# White background
color Display Background white

# Turn off axes
axes location Off

# Optional: cleaner look
display depthcue off
display shadows off
display ambientocclusion off

puts "Done. Showing only type 1 atoms colored by User (table applies to type 1 only)."

# -------- Fit view tightly to the drawn particles (minimize margins) --------
set molid [molinfo top]
set last [expr {[molinfo $molid get numframes] - 1}]
animate goto $last
display resetview

# Optional: tiny extra zoom-in (reduce margins further)
scale by 2

# Render current frame with TachyonInternal to TGA
set out "render_type1_user.tga"
render TachyonInternal $out

puts "Rendered: $out"