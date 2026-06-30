# tilt.tcl
#
### Abstract
# This file contains implementation of a tilt metric consistent 
# with the Colvars package.
#
# The primary use-case has been in restraining the orientation of
# phospholipid acyl chains during decoupling from a bulk membrane,
# but may also be used for the orientation of any well-ordered set
# of atoms (e.g. an alpha helix)
#
# Based on fit_angle.tcl by Justin Gullingsrud:
# https://www.ks.uiuc.edu/Research/vmd/mailing_list/vmd-l/att-2279/fit_angle.tcl
#
# Used in E Santiago-McRae, O Sheffer, WL Cheng, J Hénin, and GH Brannigan 2025 preprint:
# https://chemrxiv.org/doi/full/10.26434/chemrxiv-2025-t53z0-v3
#
# See also Colvars documentation: https://colvars.github.io/
#
# Authors: JH, ES

# Get the magnitude of an arbitrary vector
#
# Arguments:
# vec: a vector represented as a list
# Results:
# The length of the vector
proc get_magnitude {vec} {
    set magnitude 0.0
    foreach v $vec {
        set magnitude [expr {$magnitude + $v * $v}]
    }
    set magnitude [expr {sqrt($magnitude)}]
    
    return $magnitude
}

# Apply a unit normalization to an input vector
#
# Arguments:
# vec: an arbitrary vector
# Results:
# A unit vector of the same dimensionality and direction as the input
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

# Compute the coefficient for a one dimensional, ordered data set
# That is, compute m s.t. x_i=m*i, where i is the index of the data point
# Assumes the data points are ordered and equidistant
#
# Arguments:
# x: a one-dimensional vector of data points
# Results:
# The optimal coefficient in the least-squares sense
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

# "Slice" a vector by extracting every step'th value from
# the start point to the end of the vector.
#
# Arguments:
# vec: the vector to slice
# start: the initial index (discard values before this point)
# step: the N for selecting every Nth value
# Results:
# A new vector of length |vec|//step with only those selected values
proc reslice_vector {vec start step} {
    set idcs [ generate_range $start [llength $vec] $step]
    set sliced {}
    foreach idx $idcs {
        set val [lindex $vec $idx]
        lappend sliced $val
    }
    return $sliced
}

# Make a list of numbers from start to end counting by step
# 
# Arguments:
# start: the initial number
# end: the last number
# step: the stride
# Results:
# A vector of length (end-start)//step
# with counting numbers between start and end
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


# Fits a vector to an ordered data set
# See assumptions of lsq
#
# JH - now returns unnormalized vector because it's needed for gradient computation
#
# Arguments:
# data: a one-dimensional dataset in the form { x1 y1 z1 x2 y2 z2 ... xN yN zN }
# Results:
# An unnormalized vector that fits the data in the least squares fit*
# (*if the assumptions are valid)
proc fit_vec_to_cartesian { data } {
    set x [ reslice_vector $data 0 3 ]
    set y [ reslice_vector $data 1 3 ]
    set z [ reslice_vector $data 2 3 ]
    set vector [list [lsq $x] [lsq $y] [lsq $z] ]
    return $vector
}

# Calculate the cosine of the angle between a vector and the z axis
# 
# Arguments:
# tail: a vector (that lies along a phospholipid acyl chain in this context)
# Results:
# cosine of the angle between the vector and the z axis
proc calc_tilt { tail } {
    set unnormalized [ fit_vec_to_cartesian $tail ]
    set val [ normalize_vector $unnormalized ]
    return [lindex $val 2]
}

# Calculate the gradient of the tilt w.r.t. the z axis
# See calc_tilt
#
# Arguments:
# tail: a vector (that lies along a phospholipid acyl chain in this context)
# Results:
# the gradient of the tilt of that vector
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


