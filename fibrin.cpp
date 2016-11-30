#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <random>
#include <iomanip>
#include <cmath>
#include <map>
#include <sys/stat.h>
#include <vector>
#include <gsl/gsl_math.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_rng.h>
#include "Eigen/Dense"
#include "Eigen/SVD"

using namespace Eigen;
using namespace std;

// The model parameters
const double L1 = 1.0; // initial length of cylinder
const double D2 = 6.0; // final length of spherocylinder
const double R = 1.0; // spherocylinder radius
const double T = 10.0; // division time

const double NU = 1.0; // ambient viscosity
const double E_0 = 10.0; // the elastic modulus of cell body
const double NP = 5; // degrees of freedom

// Additional useful constants
const double A = log(2.0)/T; // growth rate of spherocylinder
// Note: pi is given by M_PI

int maxgen = 4; // how many generations does the colony grow?

class Cell {
	// One cell. Stores the coordinates and creates daughter cells when time to divide
	private:
		double x, y, z, nx, ny, nz, l; //initial node position on grid
	public:
		Cell (double, double, double, double, double, double);
		void set_pos(double mx, double my, double mz) {
			x = mx;
			y = my;
			z = mz;
		}
		void set_n(double mnx, double mny, double mnz) {
			nx = mnx;
			ny = mny;
			nz = mnz;
		}
		void set_l(double ml) {
			l = ml;
		}
		
		// Return the state of the cell
		double get_x() { return x; }
		double get_y() { return y; }
		double get_z() { return z; }
		double get_nx() { return nx; }
		double get_ny() { return ny; }
		double get_nz() { return nz; }
		double get_l() { return l; }
		
		// Create daughter cells. They are 1/2 the overall length D2 of the mother cell, so their centers must be displaced by D2/4.0
		Cell* divide_1() {
			double x2, y2, z2;
			x2 = x + get_nx()*(D2/4.0);
			y2 = y + get_ny()*(D2/4.0);
			z2 = z + get_nz()*(D2/4.0);
			return new Cell(x2, y2, z2, this->get_nx(), this->get_ny(), this->get_nz());
		}
		Cell* divide_2() {
			double x2, y2, z2;
			x2 = x - get_nx()*(D2/4.0);
			y2 = y - get_ny()*(D2/4.0);
			z2 = z - get_nz()*(D2/4.0);
			return new Cell(x2, y2, z2, this->get_nx(), this->get_ny(), this->get_nz());
		}
};

Cell::Cell(double mx, double my, double mz, double mnx, double mny, double mnz) {
	x = mx;
	y = my;
	z = mz;
	nx = mnx;
	ny = mny;
	nz = mnz;
	
	l = L1;
}

vector<Cell *> cells;

double get_overlap(Cell* cell_1, Cell* cell_2) {
	// Returns the overlap. see: http://homepage.univie.ac.at/franz.vesely/notes/hard_sticks/hst/hst.html
	double overlap;
	
	double x1, y1, z1, nx1, ny1, nz1, l1;
	double x2, y2, z2, nx2, ny2, nz2, l2;
	double x12, y12, z12, m12, u1, u2, u12, cc, xla, xmu;
	
	double rpx, rpy, rpz, rix, riy, riz;
	double h1, h2, gam, gam1, gam2, gamm, gamms, del, del1, del2, delm, delms, aa, a1, a2, risq, f1, f2;
	double gc1, dc1, gc2, dc2;
	
	x1 = cell_1->get_x();
	y1 = cell_1->get_y();
	z1 = cell_1->get_z();
	nx1 = cell_1->get_nx();
	ny1 = cell_1->get_ny();
	nz1 = cell_1->get_nz();
	l1 = cell_1->get_l() / 2.0;
	
	x2 = cell_2->get_x();
	y2 = cell_2->get_y();
	z2 = cell_2->get_z();
	nx2 = cell_2->get_nx();
	ny2 = cell_2->get_ny();
	nz2 = cell_2->get_nz();
	l2 = cell_2->get_l() / 2.0;
	
	x12 = x2 - x1;
	y12 = y2 - y1;
	z12 = z2 - z1;
	
	m12 = x12*x12 + y12*y12 + z12*z12;
	
	u1 = x12*nx1 + y12*ny1 + z12*nz1;
	u2 = x12*nx2 + y12*ny2 + z12*nz2;
	u12 = nx1*nx2 + ny1*ny2 + nz1*nz2;
	cc = 1.0 - u12*u12;
	
	// Check if parallel
	if (cc < 1e-6) {
	 	if(u1 && u2) {
			//cout << "k" << endl;
			xla = u1/2;
			xmu = -u2/2;
		}
		else {
			// lines are parallel
			//cout << "KEK" << endl;
			return sqrt(m12);
	 	}
	}
	else {
		xla = (u1 - u12*u2) / cc;
	 	xmu = (-u2 + u12*u1) / cc;
	}
	
	rpx = x12 + xmu*nx2 - xla*nx1;
	rpy = y12 + xmu*ny2 - xla*ny1;
	rpz = z12 + xmu*nz2 - xla*nz1;
	
	//Rectangle half lengths h1=L1/2, h2=L2/2	
	h1 = l1; 
	h2 = l2;

	//If the origin is contained in the rectangle, 
	//life is easy: the origin is the minimum, and 
	//the in-plane distance is zero!
	if ((xla*xla <= h1*h1) && (xmu*xmu <= h2*h2)) {
		  // Simple overlap
		  rix = 0;
		  riy = 0;
		  riz = 0;
	}
	else {
	//Find minimum of f=gamma^2+delta^2-2*gamma*delta*(e1*e2)
	//where gamma, delta are the line parameters reckoned from the intersection
	//(=lam0,mu0)

	//First, find the lines gamm and delm that are nearest to the origin:
		  gam1 = -xla - h1;
		  gam2 = -xla + h1;
		  gamm = gam1;
		  if (gam1*gam1 > gam2*gam2) { gamm=gam2; }
		  del1 = -xmu - h2;
		  del2 = -xmu + h2;
		  delm = del1;
		  if (del1*del1 > del2*del2) { delm = del2; }	

	//Now choose the line gamma=gamm and optimize delta:
		  gam = gamm;	  
		  delms = gam * u12;	  
		  aa = xmu + delms;	// look if delms is within [-xmu0+/-L/2]:
		  if (aa*aa <= h2*h2) {
		    del=delms;
		  }		// somewhere along the side gam=gamm
		  else {
	//delms out of range --> corner next to delms!
		    del = del1;
		    a1 = delms - del1;
		    a2 = delms - del2;
		    if (a1*a1 > a2*a2) {del = del2; }
		  }
		  
	// Distance at these gam, del:
		  f1 = gam*gam+del*del-2.*gam*del*u12;
		  gc1 = gam;
		  dc1 = del;
		  
	//Now choose the line delta=deltam and optimize gamma:
		  del=delm;	  
		  gamms=del*u12;	  
		  aa=xla+gamms;	// look if gamms is within [-xla0+/-L/2]:
		  if (aa*aa <= h1*h1) {
		    gam=gamms; }		// somewhere along the side gam=gamm
		  else {
	// gamms out of range --> corner next to gamms!
		    gam=gam1;
		    a1=gamms-gam1;
		    a2=gamms-gam2;
		    if (a1*a1 > a2*a2) gam=gam2;
		  }
	// Distance at these gam, del:
		  f2 = gam*gam+del*del-2.*gam*del*u12;
		  gc2 = gam;
		  dc2 = del;
		  
	// Compare f1 and f2 to find risq:
		  risq=f1;
		  rix = dc1 * nx2 - gc1 * nx1;
		  riy = dc1 * ny2 - gc1 * ny1;
		  riz = dc1 * nz2 - gc1 * nz1;
		  
		  if(f2 < f1) {
			  risq=f2;
			  rix = dc2 * nx2 - gc2 * nx1;
			  riy = dc2 * ny2 - gc2 * ny1;
			  riz = dc2 * nz2 - gc2 * nz1;
		  
		  }
	}
	
	overlap =  rpx*rpx + rpy*rpy + rpz*rpz + rix*rix + riy*riy + riz*riz;
	
	overlap = sqrt(overlap);
	
	return overlap;
}

double get_overlap_vec(Cell* cell_1, Cell* cell_2, int dof) {
	// Returns the vector of the shortest distance from cell 1 to cell 2
	// An extension of: http://homepage.univie.ac.at/franz.vesely/notes/hard_sticks/hst/hst.html
	// dof is the component of the vector (0,1,2 for cell 1, 3,4,5 for cell 2)
	double overlap;
	
	double x1, y1, z1, nx1, ny1, nz1, l1;
	double x2, y2, z2, nx2, ny2, nz2, l2;
	double x12, y12, z12, m12, u1, u2, u12, cc, xla, xmu;
	
	double rpx, rpy, rpz, rix1, riy1, riz1, rix2, riy2, riz2;
	
	double h1, h2, gam, gam1, gam2, gamm, gamms, del, del1, del2, delm, delms, aa, a1, a2, risq, f1, f2;
	double gc1, dc1, gc2, dc2;
	
	x1 = cell_1->get_x();
	y1 = cell_1->get_y();
	z1 = cell_1->get_z();
	nx1 = cell_1->get_nx();
	ny1 = cell_1->get_ny();
	nz1 = cell_1->get_nz();
	l1 = cell_1->get_l() / 2.0;
	
	x2 = cell_2->get_x();
	y2 = cell_2->get_y();
	z2 = cell_2->get_z();
	nx2 = cell_2->get_nx();
	ny2 = cell_2->get_ny();
	nz2 = cell_2->get_nz();
	l2 = cell_2->get_l() / 2.0;
	
	x12 = x2 - x1;
	y12 = y2 - y1;
	z12 = z2 - z1;
	
	m12 = x12*x12 + y12*y12 + z12*z12;
	
	u1 = x12*nx1 + y12*ny1 + z12*nz1;
	u2 = x12*nx2 + y12*ny2 + z12*nz2;
	u12 = nx1*nx2 + ny1*ny2 + nz1*nz2;
	cc = 1.0 - u12*u12;
	
	// Check if parallel
	if (cc < 1e-6) {
	 	if(u1 && u2) {
			//cout << "k" << endl;
			xla = u1/2;
			xmu = -u2/2;
		}
		else {
			// lines are parallel
			//cout << "KEK" << endl;
		  	rix1 = x1;
		  	riy1 = y1;
		  	riz1 = z1;
		  	rix2 = x2;
		  	riy2 = y2;
		  	riz2 = z2;
			
			if (dof == 0) {
				return rix1;
			} else if (dof == 1) {
				return riy1;
			} else if (dof == 2) {
				return riz1;
			} else if (dof == 3) {
				return rix2;
			} else if (dof == 4) {
				return riy2;
			} else if (dof == 5) {
				return riz2;
			}
			return 0;
			//return sqrt(m12);
	 	}
	}
	else {
		xla = (u1 - u12*u2) / cc;
	 	xmu = (-u2 + u12*u1) / cc;
	}
	
	rpx = x12 + xmu*nx2 - xla*nx1;
	rpy = y12 + xmu*ny2 - xla*ny1;
	rpz = z12 + xmu*nz2 - xla*nz1;
	
	
	//Rectangle half lengths h1=L1/2, h2=L2/2	
	h1 = l1; 
	h2 = l2;

	//If the origin is contained in the rectangle, 
	//life is easy: the origin is the minimum, and 
	//the in-plane distance is zero!
	if ((xla*xla <= h1*h1) && (xmu*xmu <= h2*h2)) {
	  	rix1 = x1 + (xla) * nx1;
	  	riy1 = y1 + (xla) * ny1;
	  	riz1 = z1 + (xla) * nz1;
	  	rix2 = x2 + (xmu) * nx2;
	  	riy2 = y2 + (xmu) * ny2;
	  	riz2 = z2 + (xmu) * nz2;
		if (dof == 0) {
			return rix1;
		} else if (dof == 1) {
			return riy1;
		} else if (dof == 2) {
			return riz1;
		} else if (dof == 3) {
			return rix2;
		} else if (dof == 4) {
			return riy2;
		} else if (dof == 5) {
			return riz2;
		}
		return 0;
	}
	else {
	//Find minimum of f=gamma^2+delta^2-2*gamma*delta*(e1*e2)
	//where gamma, delta are the line parameters reckoned from the intersection
	//(=lam0,mu0)

	//First, find the lines gamm and delm that are nearest to the origin:
		  gam1 = -xla - h1;
		  gam2 = -xla + h1;
		  gamm = gam1;
		  if (gam1*gam1 > gam2*gam2) { gamm=gam2; }
		  del1 = -xmu - h2;
		  del2 = -xmu + h2;
		  delm = del1;
		  if (del1*del1 > del2*del2) { delm = del2; }	

	//Now choose the line gamma=gamm and optimize delta:
		  gam = gamm;	  
		  delms = gam * u12;	  
		  aa = xmu + delms;	// look if delms is within [-xmu0+/-L/2]:
		  if (aa*aa <= h2*h2) {
		    del=delms;
		  }		// somewhere along the side gam=gamm
		  else {
	//delms out of range --> corner next to delms!
		    del = del1;
		    a1 = delms - del1;
		    a2 = delms - del2;
		    if (a1*a1 > a2*a2) {del = del2; }
		  }
		  
	// Distance at these gam, del:
		  f1 = gam*gam+del*del-2.*gam*del*u12;
		  gc1 = gam;
		  dc1 = del;
		  
	//Now choose the line delta=deltam and optimize gamma:
		  del=delm;	  
		  gamms=del*u12;	  
		  aa=xla+gamms;	// look if gamms is within [-xla0+/-L/2]:
		  if (aa*aa <= h1*h1) {
		    gam=gamms; }		// somewhere along the side gam=gamm
		  else {
	// gamms out of range --> corner next to gamms!
		    gam=gam1;
		    a1=gamms-gam1;
		    a2=gamms-gam2;
		    if (a1*a1 > a2*a2) gam=gam2;
		  }
	// Distance at these gam, del:
		  f2 = gam*gam+del*del-2.*gam*del*u12;
		  gc2 = gam;
		  dc2 = del;
		  
	// Compare f1 and f2 to find risq:
		  risq=f1;
		  //rix = dc1 * nx2 - gc1 * nx1;
		  //riy = dc1 * ny2 - gc1 * ny1;
		  //riz = dc1 * nz2 - gc1 * nz1;
		  rix1 = x1 + (xla + gc1) * nx1;
		  riy1 = y1 + (xla + gc1) * ny1;
		  riz1 = z1 + (xla + gc1) * nz1;
		  rix2 = x2 + (xmu + dc1) * nx2;
		  riy2 = y2 + (xmu + dc1) * ny2;
		  riz2 = z2 + (xmu + dc1) * nz2;
		  
		  if(f2 < f1) {
			  risq=f2;
			  //rix = dc2 * nx2 - gc2 * nx1;
			  //riy = dc2 * ny2 - gc2 * ny1;
			  //riz = dc2 * nz2 - gc2 * nz1;
			  rix1 = x1 + (xla + gc2) * nx1;
			  riy1 = y1 + (xla + gc2) * ny1;
			  riz1 = z1 + (xla + gc2) * nz1;
			  rix2 = x2 + (xmu + dc2) * nx2;
			  riy2 = y2 + (xmu + dc2) * ny2;
			  riz2 = z2 + (xmu + dc2) * nz2;
		  
		  }
	}
	
	if (dof == 0) {
		return rix1;
	} else if (dof == 1) {
		return riy1;
	} else if (dof == 2) {
		return riz1;
	} else if (dof == 3) {
		return rix2;
	} else if (dof == 4) {
		return riy2;
	} else if (dof == 5) {
		return riz2;
	}
	return 0;
}

double rot(double nx, double ny, double nz, double tx, double ty, double tz, double a, int dof) {
	// Rotate the vector n around axis t by an angle a (Rodrigues formula)
	// dof=0 x coordinate, 1 = y coordinate, 2 = z coordinate
	if (dof == 0) {
		return tx*(nx*tx + ny*ty + nz*tz)*(1 - cos(a)) + nx*cos(a) + (nz*ty - ny*tz)*sin(a);
	} else if (dof == 1) {
		return ty*(nx*tx + ny*ty + nz*tz)*(1 - cos(a)) + ny*cos(a) + (-(nz*tx) + nx*tz)*sin(a);
	} else if (dof == 2) {
		return tz*(nx*tx + ny*ty + nz*tz)*(1 - cos(a)) + nz*cos(a) + (ny*tx - nx*ty)*sin(a);
	}
	return 0;
}

double cross(double x1, double y1, double z1, double x2, double y2, double z2, int dof) {
	// Cross product
	if (dof == 0) {
		return -(y2*z1) + y1*z2;
	} else if (dof == 1) {
		return x2*z1 - x1*z2;
	} else if (dof == 2) {
		return -(x2*y1) + x1*y2;
	}
	return 0;
}

double force_cell(int j, int dof) {
	// Returns the net force on cell j
	
	double rx, ry, rz;
	
	double Fji, vx, vy, vz, wx, wy, wz, sx, sy, sz, sm, tx, ty, tz, t_net, overlap;
  	for (int j2 = 0; j2 != cells.size(); j2++) {
		if (j2 != j) {
  			overlap = 2*R - get_overlap(cells[j], cells[j2]);
  		  	if (overlap > 0) {
  				  //contact on cell i
  				  vx = get_overlap_vec(cells[j], cells[j2], 0);
  				  vy = get_overlap_vec(cells[j], cells[j2], 1);
  				  vz = get_overlap_vec(cells[j], cells[j2], 2);

  				  //contact on cell j
  				  wx = get_overlap_vec(cells[j], cells[j2], 3);
  				  wy = get_overlap_vec(cells[j], cells[j2], 4);
  				  wz = get_overlap_vec(cells[j], cells[j2], 5);

  				  sx = vx - wx;
  				  sy = vy - wy;
  				  sz = vz - wz;
  				  sm = sqrt(sx*sx + sy*sy + sz*sz);
				  
  				  rx = vx - cells[j]->get_x();
  				  ry = vy - cells[j]->get_y();
  				  rz = vz - cells[j]->get_z();

  				  if (sm == 0) {
  					  cout << "ERROR!" << endl;
					  sx = cells[j]->get_x() - cells[j2]->get_x();
					  sy = cells[j]->get_y() - cells[j2]->get_y();
					  sz = cells[j]->get_z() - cells[j2]->get_z();
					  sm = sqrt(sx*sx + sy*sy + sz*sz);
  				  }
		  		  
				  Fji = E_0 * pow(overlap, 3.0/2.0);
				  
  				  // Calculate torque
  				  tx += Fji*sx/sm;
  				  ty += Fji*sy/sm;
  				  tz += Fji*sz/sm;
	  			
  			}
  		}
  	}
	
	tz = 0; //2D
	
	if (dof==0) {
		return tx;
	} else if (dof==1) {
		return ty;
	} else if (dof==2) {
		return tz;
	}
	return 0;
}

double torque_cell(int j, int dof) {
	// Returns the net torque on cell j
	
	double rx, ry, rz;
	
	double Fji, vx, vy, vz, wx, wy, wz, sx, sy, sz, sm, tx, ty, tz, t_net, overlap;
  	for (int j2 = 0; j2 != cells.size(); j2++) {
		if (j2 != j) {
  			overlap = 2*R - get_overlap(cells[j], cells[j2]);
  		  	if (overlap > 0) {
  				  //contact on cell i
  				  vx = get_overlap_vec(cells[j], cells[j2], 0);
  				  vy = get_overlap_vec(cells[j], cells[j2], 1);
  				  vz = get_overlap_vec(cells[j], cells[j2], 2);

  				  //contact on cell j
  				  wx = get_overlap_vec(cells[j], cells[j2], 3);
  				  wy = get_overlap_vec(cells[j], cells[j2], 4);
  				  wz = get_overlap_vec(cells[j], cells[j2], 5);

  				  sx = vx - wx;
  				  sy = vy - wy;
  				  sz = vz - wz;
  				  sm = sqrt(sx*sx + sy*sy + sz*sz);

  				  rx = vx - cells[j]->get_x();
  				  ry = vy - cells[j]->get_y();
  				  rz = vz - cells[j]->get_z();

  				  if (sm == 0) {
  					  cout << "ERROR!" << endl;
					  sx = 0;
					  sy = 0;
					  sz = 0;
					  sm = 1;
  				  }
		  		  
				  Fji = E_0 * pow(overlap, 3.0/2.0);
				  
  				  // Calculate torque
  				  tx += cross(rx, ry, rz, Fji*sx/sm, Fji*sy/sm, Fji*sz/sm, 0);
  				  ty += cross(rx, ry, rz, Fji*sx/sm, Fji*sy/sm, Fji*sz/sm, 1);
  				  tz += cross(rx, ry, rz, Fji*sx/sm, Fji*sy/sm, Fji*sz/sm, 2);
	  			
  			}
  		}
  	}
	
	//2D motion:
	tx = 0;
	ty = 0;
	
	if (dof==0) {
		return tx;
	} else if (dof==1) {
		return ty;
	} else if (dof==2) {
		return tz;
	}
	return 0;
}

int func (double t, const double y[], double f[], void *params) {
  (void)(t); /* avoid unused parameter warning <- not sure what this note is for (see gsl library example) */
  
  // Calculate the rhs of the differential equation, dy/dt = ?
  // Contributions from: 1) ambient viscosity, 2) cell growth, 3) cell-cell pushing
  // To do: cell-cell sticking, cell-surface pushing, cell-surface adhesion, surface viscosity
  
  // The degrees of freedom are as follows:
  // f[0] = cell 1, x position
  // f[1] = cell 1, y pos
  // f[2] = cell 1, z pos
  // f[3] = cell 1, angle
  // f[4] = cell 1, length
  // f[5] cell 2, ...
  
  double overlap, Fji;
  
  double sx, sy, sz, sm;
  double rx, ry, rz, rm;
  
  double vx, vy, vz, wx, wy, wz;
  
  double tx, ty, tz, t_net;
  
  int dof = 0;
  
  // Calculate the force on cell i:
  for (int i = 0; i != cells.size(); i++) {
	  
	  f[dof] = 0;
	  f[dof+1] = 0;
	  f[dof+2] = 0;
	  
	  tx = 0;
	  ty = 0;
	  tz = 0;
	  
	  // Loop over the other cells j:
	  for (int j = 0; j != cells.size(); j++) {
		  
		  if (j != i) {
			  
			  overlap = 2*R - get_overlap(cells[i], cells[j]);
			  
			  if (overlap > 0) {
				  
				  Fji = E_0 * pow(overlap, 3.0/2.0);
				  
				  //contact on cell i
				  vx = get_overlap_vec(cells[i], cells[j], 0);
				  vy = get_overlap_vec(cells[i], cells[j], 1);
				  vz = get_overlap_vec(cells[i], cells[j], 2);
				  
				  //contact on cell j
				  wx = get_overlap_vec(cells[i], cells[j], 3);
				  wy = get_overlap_vec(cells[i], cells[j], 4);
				  wz = get_overlap_vec(cells[i], cells[j], 5);
				  
				  sx = vx - wx;
				  sy = vy - wy;
				  sz = vz - wz;
				  sm = sqrt(sx*sx + sy*sy + sz*sz);
				  
				  rx = vx - cells[i]->get_x();
				  ry = vy - cells[i]->get_y();
				  rz = vz - cells[i]->get_z();
				  
				  if (sm == 0) {
					  cout << "ERROR!" << endl;
					  sx = cells[i]->get_x() - cells[j]->get_x();
					  sy = cells[i]->get_y() - cells[j]->get_y();
					  sz = cells[i]->get_z() - cells[j]->get_z();
					  sm = sqrt(sx*sx + sy*sy + sz*sz);
				  }
				  
				  f[dof] += (1.0/NU) * (1.0/(cells[i]->get_l() + 2*R)) * Fji * sx/sm;
				  f[dof+1] += (1.0/NU) * (1.0/(cells[i]->get_l() + 2*R)) * Fji * sy/sm;
				  f[dof+2] += (1.0/NU) * (1.0/(cells[i]->get_l() + 2*R)) * Fji * sz/sm;
				  
				  // Calculate torque
				  tx += cross(rx, ry, rz, Fji*sx/sm, Fji*sy/sm, Fji*sz/sm, 0);
				  ty += cross(rx, ry, rz, Fji*sx/sm, Fji*sy/sm, Fji*sz/sm, 1);
				  tz += cross(rx, ry, rz, Fji*sx/sm, Fji*sy/sm, Fji*sz/sm, 2);
				  
			  }
		  }
	  }
	  
	  //2D motion:
	  tx = 0;
	  ty = 0;
	  
	  t_net = sqrt(tx*tx + ty*ty + tz*tz);
	  
	  // Symmetry breaking:
	  f[dof+0] += rand()/(100000*static_cast<double>(RAND_MAX));
	  f[dof+1] += rand()/(100000*static_cast<double>(RAND_MAX));
	  f[dof+2] += rand()/(100000*static_cast<double>(RAND_MAX));
	  
	  f[dof+2] = 0; //2D motion
	  
	  f[dof+3] = (1.0/NU) * pow(1.0/(cells[i]->get_l() + 2*R), 3) * t_net;
	  
	  f[dof+4] = A * (cells[i]->get_l() + 2*R);
	  
	  dof = dof + NP;
  }
  
  return GSL_SUCCESS;

}

int main(int argc, char * argv[]) {
	cout << "\n\n\n\n\n\n\n";
	cout << "Program running\n";
	
	
	// Initializations:
	int dof;
	int Ni;
	double y[5000];
	cells.clear();
	cells.reserve(3500);
	
	// Create first cell:
	cells.push_back(new Cell(0, 0, 0, 1, 0, 0));
	
	cout << "Hello world." << endl;
	
	//int trial = atoi(argv[1]);  //Optional input for seed
	int trial = 0; 
	srand ((trial+1)*time(NULL));
	
	// Output data
	string my_name = "output/biofilm";
	ofstream myfile;
	string my_mono_name = my_name+"-line.txt";
	myfile.open(my_mono_name);
	
	for (int gen = 0; gen != maxgen; gen++) {
		cout << "Gen: " << gen << endl;
		
		unsigned long dim = NP * cells.size();
		
		gsl_odeiv2_system sys = {func, nullptr, dim, nullptr};
		gsl_odeiv2_driver * d = 
			gsl_odeiv2_driver_alloc_y_new (&sys, gsl_odeiv2_step_rk8pd, 1e-6, 1e-6, 0.0);
    	
		// Set the initial conditions
		double t = 0.0, t1 = T;
		
		myfile << gen*T << " " << cells.size() << endl;
		
		dof = 0;
		for (int j = 0; j != cells.size(); j++) {
			y[dof+0] = cells[j]->get_x();
			y[dof+1] = cells[j]->get_y();
			//y[dof+2] = cells[j]->get_z(); // 2D motion only
			y[dof+2] = 0;
			y[dof+3] = 0; // initial condition for torque is always zero
			y[dof+4] = cells[j]->get_l() + 2*R;
			
			// Output stuff to file:
			myfile << y[dof+0] << " " << y[dof+1] << " " << y[dof+2] << " "
					<< cells[j]->get_nx() << " " << cells[j]->get_ny() << " " << cells[j]->get_nz()  << " "
					<< y[dof+4]-2*R << endl;
			
			dof = dof + NP;
		}
		
		// Evolve the differential equation
	    for (int i = 1; i <= 100; i++) {
			
	        double ti = i * t1 / 100.0;
			
			// Step forward!
	        int status = gsl_odeiv2_driver_apply (d, &t, ti, y);
			
			// Output the progress
			myfile << gen*T + ti << " " << cells.size() << endl;
			
			// Update the coordinates of each cell:
			dof = 0;
			for (int j = 0; j != cells.size(); j++) {
				// Rotate the vector n around t by y[dof+3]:
				// First, obtain t by summing over other cells:
				double tx, ty, tz, t_net, nxi, nyi, nzi, nm;
				
				nxi = cells[j]->get_nx();
				nyi = cells[j]->get_ny();
				nzi = cells[j]->get_nz();
				
				tx = torque_cell(j, 0);
				ty = torque_cell(j, 1);
				tz = torque_cell(j, 2);
				
				//tz = 0;
				
		  	 	t_net = sqrt(tx*tx + ty*ty + tz*tz);
				
				cells[j]->set_pos(y[dof+0],y[dof+1],y[dof+2]);
				
				if (t_net > 0) {
					cells[j]->set_n(
									rot(nxi, nyi, nzi, tx/t_net, ty/t_net, tz/t_net, y[dof+4], 0),
									rot(nxi, nyi, nzi, tx/t_net, ty/t_net, tz/t_net, y[dof+4], 1),
									rot(nxi, nyi, nzi, tx/t_net, ty/t_net, tz/t_net, y[dof+4], 2)
								   );
				}
				
				nxi = cells[j]->get_nx();
				nyi = cells[j]->get_ny();
				nzi = cells[j]->get_nz();
				nm = sqrt(nxi*nxi + nyi*nyi); // 2D only...
				
				// Normalize the orientation vector, to be sure...
				cells[j]->set_n(nxi/nm, nyi/nm, 0);
				
				cells[j]->set_l(y[dof+4] - 2*R);
				
				myfile << y[dof+0] << " " << y[dof+1] << " " << y[dof+2] << " "
						<< cells[j]->get_nx() << " " << cells[j]->get_ny() << " " << cells[j]->get_nz() << " "
						<< y[dof+4]-2*R << endl;
				
				dof = dof + NP;
			}
			
	        if (status != GSL_SUCCESS) {
		  	    printf ("error, return value=%d\n", status);
		  	    break;
		  	}
			
	    }
		
	    gsl_odeiv2_driver_free (d);
		
		// Divide the cells
		Ni = cells.size();
		
		// add next generation of daughter cells to the list
		for (int j = 0; j != Ni; j++) {
			Cell* c1 = cells[j]->divide_1();
			Cell* c2 = cells[j]->divide_2();
			cells.push_back(c1);
			cells.push_back(c2);
		}
		
		// delete the previous generation of mother cells
		cells.erase(cells.begin(), cells.begin()+Ni);
	}
	
	
	cout << "Done" << endl;
	
	return 0;
}


