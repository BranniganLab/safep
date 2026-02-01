proc get_magnitude {vec} {
    set magnitude 0.0
    foreach v $vec {
        set magnitude [expr {$magnitude + $v * $v}]
    }
    set magnitude [expr {sqrt($magnitude)}]
    
    return $magnitude
}

proc normalize_vector {vec} {
    set magnitude [get_magnitude $vec]

    if {$magnitude == 0} {
        error "Cannot normalize a zero vector"
    }

    set normalized_vec [list]
    foreach v $vec {
        lappend normalized_vec [expr {$v / $magnitude}]
    }

    return $normalized_vec
}

proc lsq { x } {
	set N [llength $x]
	set xtot 0
	set d [expr {0.5*($N-1)}]

	set i 0.0
	foreach elem $x {
		set xtot [expr {$xtot + ($i - $d) * $elem}]
		set i [expr {$i + 1.0}]
	}
	return $xtot
}

proc reslice_vector {vec start step} {
    set idcs [ generate_range $start [llength $vec] $step]
    set sliced {}
    foreach idx $idcs {
        set val [lindex $vec $idx]
        lappend sliced $val
    }
    return $sliced
}
proc generate_range {start end step} {
    if { [expr $end < $start] } {
        return {}
    }
    set result {}
    for {set i $start} {$i < $end} {incr i $step} {
        lappend result $i
    }
    return $result
}


# Takes data in the form { x1 y1 z1 x2 y2 z2 ... xN yN zN }
# and fits a vector to it
# JH - now returns unnormalized vector because it's needed for gradient computation
proc fit_vec_to_cartesian { data } {
    set x [ reslice_vector $data 0 3 ]
    set y [ reslice_vector $data 1 3 ]
    set z [ reslice_vector $data 2 3 ]
    set vector [list [lsq $x] [lsq $y] [lsq $z] ]
    return $vector
}

proc calc_tilt { tail } {
    set unnormalized [ fit_vec_to_cartesian $tail ]
    set val [ normalize_vector $unnormalized ]
    return [lindex $val 2]
}

proc calc_tilt_gradient { tail } {
    set theta [ fit_vec_to_cartesian $tail ]
    set L [ get_magnitude $theta ]
    set u [ normalize_vector $theta ]
    lassign $u ux uy uz
    set N [expr {[llength $tail] / 3}]
    set d [expr {($N-1)/2.}]
    set grad [list]

    # Could be done with vecexpr, possibly
    set Gx [expr {   - $ux * $uz  / $L}]
    set Gy [expr {   - $uy * $uz  / $L}]
    set Gz [expr {(1 - $uz * $uz) / $L}]

    for {set i 0} {$i < $N} {incr i} {
        lappend grad [expr {($i-$d) * $Gx}]
        lappend grad [expr {($i-$d) * $Gy}]
        lappend grad [expr {($i-$d) * $Gz}]
    }
    return [list $grad]
}


