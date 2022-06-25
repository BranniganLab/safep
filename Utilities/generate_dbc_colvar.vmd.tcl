# Function to generate a DBC restraint for SAFEP
# see Salari, Joseph, Lohia, Hénin and Brannigan, J. Chem. Theory Comput. 2018, 14, 12, 6560-6573
# https://arxiv.org/abs/1801.04901
#
# The objective is to reduce the opportunity for human error among the many steps.
# 

proc generate_dbc {ligspec tolerance suffix} {
    set fitting_distance 15
    
    set ligand_sel [atomselect top "noh and ($ligspec)"]
    set ligand_atomnumbers [$ligand_sel get serial]
    set ligand_numatoms [llength $ligand_atomnumbers]
    set fittinggroup_sel [atomselect top "alpha_helix and name CA and within $fitting_distance of ($ligspec)"]
    set fittinggroup_atomnumbers [$fittinggroup_sel get serial]
    set fittinggroup_numatoms [llength $fittinggroup_atomnumbers]

    puts "For the ligand \"$ligspec\" these are the atoms: $ligand_atomnumbers"

    set all [atomselect top all]
    $all set occupancy 0
    $fittinggroup_sel set occupancy 1

    set ref_fname "rest_ref.${suffix}.pdb"
    $all writepdb $ref_fname
    set colvars_fname "safep_rest.${suffix}.colvars"
    set fd [open $colvars_fname w]
    set molname [molinfo top get name]
    puts $fd "colvar {
    # A distance to bound configuration (DBC) coordinate for ligand binding restraints.
    # See Salari, Joseph, Lohia, Hénin and Brannigan, J. Chem. Theory Comput. 2018, 14, 12, 6560-6573
    # https://arxiv.org/abs/1801.04901

    # Molecular system: $molname
    # Ligand is defined as: $ligspec

    name dbc

    rmsd {
        # Reference coordinates for ligand RMSD calculation, subject to moving frame of reference
        # as defined below
        refpositionsfile $ref_fname

        atoms {
            # Define the $ligand_numatoms ligand atoms used for RMSD calculation
            atomNumbers $ligand_atomnumbers

            # Define the moving frame of reference as fit to atoms in the binding site
            centerReference yes
            rotateReference yes
            fittingGroup {
                # These are alpha carbons in alpha helices,
                # within $fitting_distance A of \"$ligspec\"
                atomNumbers $fittinggroup_atomnumbers
            }
            # Reference coordinates for those $fittinggroup_numatoms binding site atoms
            refPositionsFile $ref_fname
        }
    }
}

harmonicWalls {
    name dbc_wall
    colvars dbc
    upperWalls $tolerance
    upperWallConstant 100.0
}
";
    close $fd
    puts "Wrote DBC colvars file: $colvars_fname"
    puts "Note that it contains ONLY the DBC and not a spherical volume restraint!"
}