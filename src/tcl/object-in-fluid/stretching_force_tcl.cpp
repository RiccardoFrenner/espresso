/*
  Copyright (C) 2012,2013,2014 The ESPResSo project
  
  This file is part of ESPResSo.
  
  ESPResSo is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  
  ESPResSo is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>. 
*/

#include "utils.hpp"
#include "tcl/parser.hpp"
#include "stretching_force_tcl.hpp"
#include "object-in-fluid/stretching_force.hpp"

/** \file stretching_force.hpp
 *  Routines to calculate the STRETCHING_FORCE Energy or/and STRETCHING_FORCE force 
 *  for a particle pair. (Dupin2007)
 *  \ref forces.cpp
*/

/************************************************************/

/// parse parameters for the stretching_force potential
int tclcommand_inter_parse_stretching_force(Tcl_Interp *interp, int bond_type, int argc, char **argv)
{
  double ks, r0;

  if (argc != 3) {
    Tcl_AppendResult(interp, "stretching_force needs 2 parameters: "
		     "<r0> <ks>", (char *) NULL);
    return TCL_ERROR;
  }

  if ((! ARG_IS_D(1, r0)) || (! ARG_IS_D(2, ks)))
    {
      Tcl_AppendResult(interp, "stretching_force needs 2 DOUBLE parameters: "
		       "<r0> <ks>", (char *) NULL);
      return TCL_ERROR;
    }
  
  CHECK_VALUE(stretching_force_set_params(bond_type, r0, ks), "bond type must be nonnegative");
}

int tclprint_to_result_stretchingforceIA(Tcl_Interp *interp, Bonded_ia_parameters *params){
	char buffer[TCL_DOUBLE_SPACE];
	Tcl_PrintDouble(interp, params->p.stretching_force.r0, buffer);
    Tcl_AppendResult(interp, "STRETCHING_FORCE ", buffer, " ", (char *) NULL);
    Tcl_PrintDouble(interp, params->p.stretching_force.ks, buffer);
    Tcl_AppendResult(interp, " ", buffer, (char *) NULL);
    return (TCL_OK);
}

//#endif

