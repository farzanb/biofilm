#include <iostream>
#include <complex>
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
const double R = 1.0; // spherocylinder radius
const double L1 = 2.5; // initial length of cylinder

const double L_div = 2*(L1 + 2*R); // Simple div
//const double L_div = 2*(L1); // Smart div

const double A = 0.05;	// growth rate
const double T = 80;

const double E_0 = 1.0; // the elastic modulus of the ground /// the overall stickyness of the ground
const double A_0 = E_0/1000.0; // surface force
const double C_0 = 1000.0;

const double E_1 = E_0*100.0;
const double A_1 = E_1/1000.0;

//const double NU_0 = 1.0; // ambient viscosity
//const double NU_1 = 10.0; //surface viscosity
const double NU_0 = 0.1 * 20.0 * A / (E_0 * R * R);
const double NU_1 = NU_0 * 10.0;
const double T_0 = 1.0 / 12.0;

const double DOF_GROW = 7; // degrees of freedom (growing)
const double DOF_DIV = 13; // degrees of freedom (division)
const int MAX_DOF = 5000;

const double tstep = 0.1; // for output porpoises only

// Additional useful constants
// Note: pi is given by M_PI

const double xy = 1; // Confine to xy plane?
const double xz = 0; // Confine to xz plane?
//const double noise_level = 1e-6;
const double noise_level = 1e-6;
const double h0 = 1e-6; // initial accuracy of diff. eq. solver

struct overlapVec {
    double rix1, riy1, riz1;
	double rix2, riy2, riz2;
	double overlap;
};
struct myVec {
	double x, y, z;
};

double cot(double i) { return(1 / tan(i)); }
double csc(double i) { return(1 / sin(i)); }

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

class Cell {
	// One cell. Stores the coordinates and creates daughter cells when time to divide
	private:
		double x, y, z, nx, ny, nz, l; //initial node position on grid
	public:
		Cell (double, double, double, double, double, double, double);
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
};

Cell::Cell(double mx, double my, double mz, double mnx, double mny, double mnz, double ml) {
	x = mx;
	y = my;
	z = mz;
	nx = mnx;
	ny = mny;
	nz = mnz;
	
	l = ml;
}

class Body {
	// One membrane-enclosed entity. When d = 0, contains a single cell.
	// When d > 0, contains two cells
	private:
		double x, y, z; 	// Center of the cell (during elongation); "center of mass" location (during division)
		double nx1, ny1, nz1, nx2, ny2, nz2, l, d, Lf;
		double x1, y1, z1, x2, y2, z2;
		int dividing;
		Cell* self_cell;
		Cell* d1_cell;
		Cell* d2_cell;
	public:
		Body(double, double, double, double, double, double, double);
		void set_pos(double mx, double my, double mz) {
			x = mx;
			y = my;
			z = mz;
		}
		void set_x1(double mx, double my, double mz) {
			x1 = mx;
			y1 = my;
			z1 = mz;
		}
		void set_x2(double mx, double my, double mz) {
			x2 = mx;
			y2 = my;
			z2 = mz;
		}
		void set_n1(double mnx, double mny, double mnz) {
			nx1 = mnx;
			ny1 = mny;
			nz1 = mnz;
		}
		void set_n2(double mnx, double mny, double mnz) {
			nx2 = mnx;
			ny2 = mny;
			nz2 = mnz;
		}
		void set_l(double ml) {
			l = ml;
		}
		void set_d(double md) {
			d = md;
		}
		
		// Return the state of the cell
		double get_x() { return x; }
		double get_y() { return y; }
		double get_z() { return z; }
		double get_x1() { return x1; }
		double get_y1() { return y1; }
		double get_z1() { return z1; }
		double get_x2() { return x2; }
		double get_y2() { return y2; }
		double get_z2() { return z2; }
		double get_nx1() { return nx1; }
		double get_ny1() { return ny1; }
		double get_nz1() { return nz1; }
		double get_nx2() { return nx2; }
		double get_ny2() { return ny2; }
		double get_nz2() { return nz2; }
		double get_l() { return l; }
		double get_d() {
			if (d < 2*R) {
				return d;
			} else {
				//cout << "2*R: " << 2*R << endl;
				return (2*R);
			}
			//return d;
		}
		double get_Lf() { return Lf; }
		double get_vol() {
			//cout << l*M_PI*pow(R,2) + (4*M_PI*pow(R,3))/3. << " " << 8.0*M_PI*R*R*R/3.0 + 2*M_PI*R*R*L1 - M_PI*(4*R+d)*(2*R-d)*(2*R-d)/12.0 << endl;
			if (d == 0) {
				return l*M_PI*pow(R,2) + (4*M_PI*pow(R,3))/3.;
			} else {
				return 8.0*M_PI*R*R*R/3.0 + 2*M_PI*R*R*l - M_PI*(4*R+d)*(2*R-d)*(2*R-d)/12.0;
			}
		}
		double get_dividing() {
			return dividing;
		}
		void start_division() {
			double offset;
			dividing = 1;
			d = 0;
			offset = l/4.0;
			//d = ml - Lf;
			//offset = d/2.0 + Lf/4.0;
			x1 = x + (offset)*nx1;
			y1 = y + (offset)*ny1;
			z1 = z + (offset)*nz1;
			x2 = x - (offset)*nx1;
			y2 = y - (offset)*ny1;
			z2 = z - (offset)*nz1;
			nx2 = nx1;
			ny2 = ny1;
			nz2 = nz1;
			
			//cout << x2 << " " << y2 << " " << z2 << endl;
		}
		/*
		double get_theta_axis(int dof) {
			double t0, t1, t2, tn;
			t0 = cross(nx1, ny1, nz1, nx2, ny2, nz2, 0);
			t1 = cross(nx1, ny1, nz1, nx2, ny2, nz2, 1);
			t2 = cross(nx1, ny1, nz1, nx2, ny2, nz2, 2);
			
			tn = sqrt(t0*t0 + t1*t1 + t2*t2);
			
			if (tn < 1e-6) {
				t0 = cross(nx1+1, ny1, nz1, nx2, ny2, nz2, 0);
				t1 = cross(nx1+1, ny1, nz1, nx2, ny2, nz2, 1);
				t2 = cross(nx1+1, ny1, nz1, nx2, ny2, nz2, 2);
			}
			
			tn = sqrt(t0*t0 + t1*t1 + t2*t2);
			
			if (tn < 1e-6) {
				t0 = cross(nx1, ny1+1, nz1, nx2, ny2, nz2, 0);
				t1 = cross(nx1, ny1+1, nz1, nx2, ny2, nz2, 1);
				t2 = cross(nx1, ny1+1, nz1, nx2, ny2, nz2, 2);
			}
			
			tn = sqrt(t0*t0 + t1*t1 + t2*t2);
			
			if (dof == 0) {
				return t0/tn;
			}
			else if (dof == 1) {
				return t1/tn;
			}
			else if (dof == 2) {
				return t2/tn;
			} else {
				cout << "THETA AXIS ERROR" << endl;
				return -1;
			}
		}
		double get_n1_axis(int dof) {
			double t0, t1, t2, a0, a1, a2, an;
			t0 = this->get_theta_axis(0);
			t1 = this->get_theta_axis(1);
			t2 = this->get_theta_axis(2);
			
			a0 = cross(nx1, ny1, nz1, t0, t1, t2, 0);
			a1 = cross(nx1, ny1, nz1, t0, t1, t2, 1);
			a2 = cross(nx1, ny1, nz1, t0, t1, t2, 2);
			
			an = sqrt(a0*a0 + a1*a1 + a2*a2);
			
			if (dof == 0) {
				return a0/an;
			}
			else if (dof == 1) {
				return a1/an;
			}
			else if (dof == 2) {
				return a2/an;
			} else {
				cout << "N1 AXIS ERROR" << endl;
				return -2;
			}
		}
		double get_n2_axis(int dof) {
			double t0, t1, t2, a0, a1, a2, an;
			t0 = this->get_theta_axis(0);
			t1 = this->get_theta_axis(1);
			t2 = this->get_theta_axis(2);
			
			a0 = cross(nx2, ny2, nz2, t0, t1, t2, 0);
			a1 = cross(nx2, ny2, nz2, t0, t1, t2, 1);
			a2 = cross(nx2, ny2, nz2, t0, t1, t2, 2);
			
			an = sqrt(a0*a0 + a1*a1 + a2*a2);
			
			if (dof == 0) {
				return a0/an;
			}
			else if (dof == 1) {
				return a1/an;
			}
			else if (dof == 2) {
				return a2/an;
			} else {
				cout << "N1 AXIS ERROR" << endl;
				return -2;
			}
		}
		*/
		double get_sf_axis(int dof) {
			double t0, t1, t2, a0, a1, a2, an;
			
			a0 = cross(0, 0, 1, nx1, ny1, nz1, 0);
			a1 = cross(0, 0, 1, nx1, ny1, nz1, 1);
			a2 = cross(0, 0, 1, nx1, ny1, nz1, 2);
			
			an = sqrt(a0*a0 + a1*a1 + a2*a2);
			
			if (dof == 0) {
				return a0/an;
			}
			else if (dof == 1) {
				return a1/an;
			}
			else if (dof == 2) {
				return a2/an;
			} else {
				cout << "SF AXIS ERROR" << endl;
				return 0;
			}
		}
		double get_spring_const() {
			double kappa;
			
			if (d > 2*R) {
				return 0;
			}
			
			//return 0;
			
			//double cutoff = 1000000000;
			//if (d == 0) {
				//kappa = cutoff;
				//} else {
				//kappa = (2*R - d) / (2*R*d);
				//}
			kappa = (2*R - d) / (2*R*d);
				
			//if (kappa > cutoff) {
			//	kappa = cutoff;
			//}
			
			//kappa = kappa;
			return kappa;
		}
		Cell* get_self() {
			self_cell->set_pos(x, y, z);
			self_cell->set_n(this->get_nx1(), this->get_ny1(), this->get_nz1());
			self_cell->set_l(l);
			return self_cell;
		}
		Cell* get_daughter(int c) {
			if (c == 1) {
				d1_cell->set_pos(x1, y1, z1);
				d1_cell->set_n(this->get_nx1(), this->get_ny1(), this->get_nz1());
				d1_cell->set_l(l/2.0);
				return d1_cell;
			}
			if (c == 2) {
				d2_cell->set_pos(x2, y2, z2);
				d2_cell->set_n(this->get_nx2(), this->get_ny2(), this->get_nz2());
				d2_cell->set_l(l/2.0);
				return d2_cell;
			} else {
				cout << "RETRIEVE DAUGHTER ERROR" << endl;
				return new Cell(0, 0, 0, 1, 0, 0, L1);
			}
		}
};

Body::Body(double mx, double my, double mz, double mnx, double mny, double mnz, double ml) {
	x = mx;
	y = my;
	z = mz;
	nx1 = mnx;
	ny1 = mny;
	nz1 = mnz;
	
	//nx2 = nx1;
	//ny2 = ny2;
	//nz2 = nz2;
	
	l = ml;
	
	Lf = L_div;
	
	self_cell = new Cell(x, y, z, nx1, ny1, nz1, l);
	d1_cell = new Cell(x, y, z, nx1, ny1, nz1, l);
	d2_cell = new Cell(x, y, z, nx1, ny1, nz1, l);
	
	dividing = 0;
	//d = 0; // During the elongation phase, the two daughters are conjoined.
}

overlapVec get_overlap_Struct(Cell* cell_1, Cell* cell_2) {
	// Returns the vector of the shortest distance from cell 1 to cell 2
	// An extension of: http://homepage.univie.ac.at/franz.vesely/notes/hard_sticks/hst/hst.html
	// dof is the component of the vector (0,1,2 for cell 1, 3,4,5 for cell 2)
	double overlap;
	
	overlapVec answer;
	
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
		  	
			//rix1 = x1;
		  	//riy1 = y1;
		  	//riz1 = z1;
		  	//rix2 = x2;
		  	//riy2 = y2;
		  	//riz2 = z2;
			
			answer.rix1 = x1;
			answer.riy1 = y1;
			answer.riz1 = z1;
			answer.rix2 = x2;
			answer.riy2 = y2;
			answer.riz2 = z2;
			
			return answer;
			
			/*
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
			return 0;*/
			//return sqrt(m12);
	 	}
	}
	else {
		xla = (u1 - u12*u2) / cc;
	 	xmu = (-u2 + u12*u1) / cc;
	}
	
	//rpx = x12 + xmu*nx2 - xla*nx1;
	//rpy = y12 + xmu*ny2 - xla*ny1;
	//rpz = z12 + xmu*nz2 - xla*nz1;
	
	
	//Rectangle half lengths h1=L1/2, h2=L2/2	
	h1 = l1; 
	h2 = l2;

	//If the origin is contained in the rectangle, 
	//life is easy: the origin is the minimum, and 
	//the in-plane distance is zero!
	if ((xla*xla <= h1*h1) && (xmu*xmu <= h2*h2)) {
	  	answer.rix1 = x1 + (xla) * nx1;
	  	answer.riy1 = y1 + (xla) * ny1;
	  	answer.riz1 = z1 + (xla) * nz1;
	  	answer.rix2 = x2 + (xmu) * nx2;
	  	answer.riy2 = y2 + (xmu) * ny2;
	  	answer.riz2 = z2 + (xmu) * nz2;
		
		return answer;
		
		/*
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
		return 0;*/
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
		  answer.rix1 = x1 + (xla + gc1) * nx1;
		  answer.riy1 = y1 + (xla + gc1) * ny1;
		  answer.riz1 = z1 + (xla + gc1) * nz1;
		  answer.rix2 = x2 + (xmu + dc1) * nx2;
		  answer.riy2 = y2 + (xmu + dc1) * ny2;
		  answer.riz2 = z2 + (xmu + dc1) * nz2;
		  
		  if(f2 < f1) {
			  risq=f2;
			  //rix = dc2 * nx2 - gc2 * nx1;
			  //riy = dc2 * ny2 - gc2 * ny1;
			  //riz = dc2 * nz2 - gc2 * nz1;
			  answer.rix1 = x1 + (xla + gc2) * nx1;
			  answer.riy1 = y1 + (xla + gc2) * ny1;
			  answer.riz1 = z1 + (xla + gc2) * nz1;
			  answer.rix2 = x2 + (xmu + dc2) * nx2;
			  answer.riy2 = y2 + (xmu + dc2) * ny2;
			  answer.riz2 = z2 + (xmu + dc2) * nz2;
		  
		  }
	}
	
	return answer;
	/*
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
	return 0;*/
}

vector<Body *> cells;

myVec cell_cell_force(double r0, Cell* cell_1, Cell* cell_2) {
	// Return the force acting on cell 1 caused by cell 2
	// dof = 0 (x coord), dof = 1 (y), dof = 2 (z)
	
	double rx, ry, rz;
	
	double Fji, vx, vy, vz, wx, wy, wz, sx, sy, sz, sm, tx, ty, tz, t_net, overlap;
	
	//double Fx, Fy, Fz;
	myVec F;
	
	overlapVec c12 = get_overlap_Struct(cell_1, cell_2);
	
	overlap = r0 - sqrt ( (c12.rix1 - c12.rix2)*(c12.rix1 - c12.rix2) +
					 (c12.riy1 - c12.riy2)*(c12.riy1 - c12.riy2) +
				     (c12.riz1 - c12.riz2)*(c12.riz1 - c12.riz2) );
					
	//overlap = r0 - get_overlap(cell_1, cell_2);
	
  	if (overlap > 0) {
		  //contact on cell i
		  //vx = get_overlap_vec(cell_1, cell_2, 0);
		  //vy = get_overlap_vec(cell_1, cell_2, 1);
		  //vz = get_overlap_vec(cell_1, cell_2, 2);
		  vx = c12.rix1;
		  vy = c12.riy1;
		  vz = c12.riz1;
	  
		  //contact on cell j
		  //wx = get_overlap_vec(cell_1, cell_2, 3);
		  //wy = get_overlap_vec(cell_1, cell_2, 4);
		  //wz = get_overlap_vec(cell_1, cell_2, 5);
		  wx = c12.rix2;
		  wy = c12.riy2;
		  wz = c12.riz2;
		  
		  sx = vx - wx;
		  sy = vy - wy;
		  sz = vz - wz;
		  sm = sqrt(sx*sx + sy*sy + sz*sz);
		  
		  rx = vx - cell_1->get_x();
		  ry = vy - cell_1->get_y();
		  rz = vz - cell_1->get_z();

		  if (sm == 0) {
			  //cout << "SM ERROR! CELLS ARE ON TOP OF EACH OTHER: " << sm << endl;
			  sx = cell_1->get_x() - cell_2->get_x();
			  sy = cell_1->get_y() - cell_2->get_y();
			  sz = cell_1->get_z() - cell_2->get_z();
			  sm = sqrt(sx*sx + sy*sy + sz*sz);
			  
			  //cout << "SM ERROR! CELLS ARE ON TOP OF EACH OTHER: " << sx << " " << sy << " " << sz << " " << sm << " " << E_0 * pow(overlap, 3.0/2.0) << " " << r0 << " " << get_overlap(cell_1, cell_2) << endl;
			  cout << "ERROR! CELLS ARE ON TOP OF EACH OTHER" << endl;
			  //cout << cell_1->get_x() << " " << cell_2->get_x() << endl;
			  //cout << cell_1->get_y() << " " << cell_2->get_y() << endl;
			  //cout << cell_1->get_z() << " " << cell_2->get_z() << endl;
			  //cout << "Cell length: " << cell_1->get_l() << endl;
			  //cout << endl;
			  
		  }
  		  
		  
		  Fji = E_1 * pow(overlap, 3.0/2.0) - A_1;
		  
		  //if (sm == 0) {
		  //	cout << "FJI: " << Fji << endl;
		  //}
		  
		  // Calculate force
		  F.x = Fji*sx/sm;
		  F.y = Fji*sy/sm;
		  F.z = Fji*sz/sm;
		
	}
	else {
		F.x = 0; F.y = 0; F.z = 0;
	}
	
	return F;
	
	/*
	if (dof==0) {
		return Fx;
	} else if (dof==1) {
		return Fy;
	} else if (dof==2) {
		return Fz;
	} else {
		cout << "CELL CELL FORCE ERROR" << endl;
		return 0;
	}*/
}

myVec cell_cell_torque(double r0, Cell* cell_1, Cell* cell_2) {
	// Return the torque acting on cell 1 caused by cell 2
	// dof = 0 (x coord), dof = 1 (y), dof = 2 (z)
	
	double rx, ry, rz;
	
	double Fji, vx, vy, vz, wx, wy, wz, sx, sy, sz, sm, tx, ty, tz, t_net, overlap;
	
	myVec t;
	
	overlapVec c12 = get_overlap_Struct(cell_1, cell_2);
	
	overlap = r0 - sqrt ( (c12.rix1 - c12.rix2)*(c12.rix1 - c12.rix2) +
					 (c12.riy1 - c12.riy2)*(c12.riy1 - c12.riy2) +
				     (c12.riz1 - c12.riz2)*(c12.riz1 - c12.riz2) );
	//overlap = r0 - get_overlap(cell_1, cell_2);
	
  	if (overlap > 0) {
		  //contact on cell i
		  //vx = get_overlap_vec(cell_1, cell_2, 0);
		  //vy = get_overlap_vec(cell_1, cell_2, 1);
		  //vz = get_overlap_vec(cell_1, cell_2, 2);
		  vx = c12.rix1;
		  vy = c12.riy1;
		  vz = c12.riz1;
  
		  //contact on cell j
		  //wx = get_overlap_vec(cell_1, cell_2, 3);
		  //wy = get_overlap_vec(cell_1, cell_2, 4);
		  //wz = get_overlap_vec(cell_1, cell_2, 5);
		  wx = c12.rix2;
		  wy = c12.riy2;
		  wz = c12.riz2;

		  sx = vx - wx;
		  sy = vy - wy;
		  sz = vz - wz;
		  sm = sqrt(sx*sx + sy*sy + sz*sz);

		  rx = vx - cell_1->get_x();
		  ry = vy - cell_1->get_y();
		  rz = vz - cell_1->get_z();
		  
		  if (sm == 0) {
			  cout << "TORQUE ERROR!" << endl;
			  t.x = 0;
			  t.y = 0;
			  t.z = 0;
			  //cout << "WOW" << endl;
		  } else {
			  
			  Fji = E_1 * pow(overlap, 3.0/2.0) - A_1;
	  
			  // Calculate torque
			  t.x = cross(rx, ry, rz, Fji*sx/sm, Fji*sy/sm, Fji*sz/sm, 0);
			  t.y = cross(rx, ry, rz, Fji*sx/sm, Fji*sy/sm, Fji*sz/sm, 1);
			  t.z = cross(rx, ry, rz, Fji*sx/sm, Fji*sy/sm, Fji*sz/sm, 2);
			  
		  }
	}
	else {
		t.x = 0; t.y = 0; t.z = 0;
	}
	
	return t;
	/*
	if (dof==0) {
		return t.x;
	} else if (dof==1) {
		return t.y;
	} else if (dof==2) {
		return t.z;
	} else {
		cout << "TORQUE RETURN ERROR!!!" << endl;
		return -999;
	}*/
}

double cell_surface_force(Cell* cell_1) {
	// Return the surface force acting on cell 1
	
	double t;
	
	double nzi = cell_1->get_nz();
	if (nzi < -1) {
		nzi = -1;
	}
	if (nzi > 1) {
		nzi = 1;
	}
	
	if (nzi > 0) {
		t = asin(nzi);
	} else {
		t = asin(-nzi);
	}
	
	if (t == 0) {
		t = 1e-10;
	}
	
	double l = cell_1->get_l();
	double z = cell_1->get_z();
	
	double h1 = z + l*nzi/2.0;
	double h2 = z - l*nzi/2.0;
	
	
	//double answer;
	
	if (h1 > h2) {
		h1 = h2;
	}
	
	if (h1 > R) {
		//cout << "NO contact" << endl;
		return 0;
	} else if (h1 > (R - l*sin(t))) {
		// Partial contact
		//cout << "PARTIAl: " << t << endl;
		return -(A_0*C_0*sin(t) + A_0*cos(t)*cot(t)*sqrt(R - z + (l*sin(t))/2.) + 
   4*(-(C_0*E_0*sin(t)*pow(R - z + (l*sin(t))/2.,1.5)) - 
      (E_0*cos(t)*cot(t)*pow(2*R - 2*z + l*sin(t),2))/4.));
	} else if (h1 > 0) {
		// Full contact
		//cout << "FUll CONTACT: " << cell_1->get_z() << " " << nzi << " " << h1 << " " << t << endl;
		return -(2*E_0*l*(-R + z)*pow(cos(t),2) - (A_0*cos(t)*cot(t)*
      (sqrt(2*R - 2*z - l*sin(t)) - sqrt(2*R - 2*z + l*sin(t))))/sqrt(2) + 
   (C_0*E_0*sin(t)*(-((pow(l,2) + 8*pow(R - z,2))*
           (sqrt(2*R - 2*z - l*sin(t)) - sqrt(2*R - 2*z + l*sin(t)))) + 
        pow(l,2)*cos(2*t)*(sqrt(2*R - 2*z - l*sin(t)) - sqrt(2*R - 2*z + l*sin(t))) - 
        8*l*(R - z)*sin(t)*(sqrt(2*R - 2*z - l*sin(t)) + sqrt(2*R - 2*z + l*sin(t)))))/
    (4.*sqrt(4*R - 4*z - 2*l*sin(t))*sqrt(2*R - 2*z + l*sin(t))));
	} else {
		cout << "ERROR: Cell is below the ground!" << endl;
		
		return 0;
	}
	
}

double cell_surface_torque(Cell* cell_1) {
	// Return the surface torque acting on cell 1
	
	//return 0;
	
	double t;
	
	double nzi = cell_1->get_nz();
	if (nzi < -1) {
		nzi = -1;
	}
	if (nzi > 1) {
		nzi = 1;
	}
	
	if (nzi > 0) {
		t = asin(nzi);
	} else {
		t = asin(-nzi);
	}
	if (t == 0) {
		t = 1e-10;
	}
	
	double l = cell_1->get_l();
	double z = cell_1->get_z();
	
	double h1 = z + l*nzi/2.0;
	double h2 = z - l*nzi/2.0;
	
	double pf = -1;
	
	if (h1 > h2) {
		h1 = h2;
		pf = 1;
	}
	
	if (h1 > R) {
		return 0;
	} else if (h1 > (R - l*sin(t))) {
		// Partial contact
		return pf*(-(A_0*C_0*cos(t)*(R - z + l*sin(t))) - 
   (A_0*cot(t)*csc(t)*sqrt(R - z + (l*sin(t))/2.)*
      (-24*R + 24*z + 8*(R - z)*cos(2*t) - 11*l*sin(t) + 5*l*sin(3*t)))/24. + 
   4*((C_0*E_0*cos(t)*pow(2*R - 2*z + l*sin(t),1.5)*(4*R - 4*z + 7*l*sin(t)))/
       (20.*sqrt(2)) + (E_0*cot(t)*csc(t)*pow(2*R - 2*z + l*sin(t),2)*
         (-3*R + 3*z + (R - z)*cos(2*t) - l*sin(t) + l*sin(3*t)))/24.));
	} else if (h1 > 0) {
		// Full contact
		return pf*((C_0*E_0*cos(t)*(16*pow(R - z,3)*(sqrt(2*R - 2*z - l*sin(t)) - 
           sqrt(2*R - 2*z + l*sin(t))) + 
        32*pow(l,2)*(R - z)*pow(sin(t),2)*
         (sqrt(2*R - 2*z - l*sin(t)) - sqrt(2*R - 2*z + l*sin(t))) + 
        44*l*pow(R - z,2)*sin(t)*(sqrt(2*R - 2*z - l*sin(t)) + 
           sqrt(2*R - 2*z + l*sin(t))) + 
        7*pow(l,3)*pow(sin(t),3)*
         (sqrt(2*R - 2*z - l*sin(t)) + sqrt(2*R - 2*z + l*sin(t)))))/
    (20.*sqrt(4*R - 4*z - 2*l*sin(t))*sqrt(2*R - 2*z + l*sin(t))) - A_0*C_0*l*sin(2*t) - 
   E_0*l*pow(R - z,2)*sin(2*t) - (A_0*cot(t)*csc(t)*
      (24*(R - z)*(sqrt(2*R - 2*z - l*sin(t)) - sqrt(2*R - 2*z + l*sin(t))) - 
        8*(R - z)*cos(2*t)*(sqrt(2*R - 2*z - l*sin(t)) - sqrt(2*R - 2*z + l*sin(t))) + 
        l*(sqrt(2*R - 2*z - l*sin(t)) + sqrt(2*R - 2*z + l*sin(t)))*
         (-11*sin(t) + 5*sin(3*t))))/(24.*sqrt(2)) + (E_0*pow(l,3)*sin(4*t))/24.);
	} else {
		cout << "ERROR: Torque is below the ground!" << endl;
		return 0;
	}
	
}

double cell_surface_viscosity(Cell* cell_1) {
	// Return the surface viscosity for cell 1
	
	double t;
	
	double nzi = cell_1->get_nz();
	if (nzi < -1) {
		nzi = -1;
	}
	if (nzi > 1) {
		nzi = 1;
	}
	
	if (nzi > 0) {
		t = asin(nzi);
	} else {
		t = asin(-nzi);
	}
	
	if (t == 0) {
		t = 1e-10;
	}
	
	double l = cell_1->get_l();
	double z = cell_1->get_z();
	
	double h1 = z + l*nzi/2.0;
	double h2 = z - l*nzi/2.0;
	
	if (h1 > h2) {
		h1 = h2;
	}
	
	if (h1 > R) {
		//cout << "NO contact" << endl;
		return 0;
	} else if (h1 > (R - l*sin(t))) {
		// Partial contact
		return NU_1 * ((-h1 + R) + (2*pow(-h1 + R,1.5)*cot(t))/3.);
	} else if (h1 > 0) {
		// Full contact
		return NU_1 * (l*sin(t) + (2*cot(t)*(l*sin(t)*sqrt(-h1 + R - l*sin(t)) - 
        (h1 - R)*(sqrt(-h1 + R) - sqrt(-h1 + R - l*sin(t)))))/3.);
	} else {
		cout << "ERROR: Cell is below the ground!" << endl;
		
		return 0;
	}
}

myVec net_force_elon(int i) {
	// Returns the net force on cells[i] during elongation
	
	// "i" is between 0 and the length of vector "cells"
	// dof is 0, 1 or 2 (for x, y, or z)
	
	//double Fx, Fy, Fz;
	
	//Fx = 0;
	//Fy = 0;
	//Fz = 0;
	myVec F, Fi;
	F.x = 0;
	F.y = 0;
	F.z = 0;
	
	// Loop over all bodies
  	for (int j = 0; j != cells.size(); j++) {
		
		// Calculate forces due to other bodies:
		if (i != j) {
			
			Fi = cell_cell_force(2*R, cells[i]->get_self(), cells[j]->get_self());
			
			F.x += Fi.x;
			F.y += Fi.y;
			F.z += Fi.z;
			
			//F.x += cell_cell_force(2*R, cells[i]->get_self(), cells[j]->get_self());
			//F.y += cell_cell_force(2*R, cells[i]->get_self(), cells[j]->get_self());
			//F.z += cell_cell_force(2*R, cells[i]->get_self(), cells[j]->get_self());
			
		}
  	}
	
	// Calculate surface force:
	F.z += cell_surface_force(cells[i]->get_self());
	
	
	if (xy == 1) {
		F.z = 0;
	}
	if (xz == 1) {
		F.y = 0;
	}
	
	return F;
	
	/*
	if (dof==0) {
		return Fx;
	} else if (dof==1) {
		return Fy;
	} else if (dof==2) {
		return Fz;
	}
	cout << "NET FORCE ELON ERROR" << endl;
	return 0;*/
	
}

myVec net_torque_elon(int i) {
	// Returns the net torque on cells[i] during elongation
	
	// "i" is between 0 and the length of vector "cells"
	// dof is 0, 1 or 2 (for x, y, or z)
	
	//double tx, ty, tz;
	
	
	myVec t, ti;
	t.x = 0;
	t.y = 0;
	t.z = 0;
	// Loop over all bodies
  	for (int j = 0; j != cells.size(); j++) {
		
		// Calculate torques due to other bodies:
		if (i != j) {
			
			ti = cell_cell_torque(2*R, cells[i]->get_self(), cells[j]->get_self());
			
			t.x += ti.x;
			t.y += ti.y;
			t.z += ti.z;
			
			//t.x += cell_cell_torque(2*R, cells[i]->get_self(), cells[j]->get_self());
			//t.y += cell_cell_torque(2*R, cells[i]->get_self(), cells[j]->get_self());
			//t.z += cell_cell_torque(2*R, cells[i]->get_self(), cells[j]->get_self());
			
		}
		
  	}
	
	// Surface torque
	t.x += cell_surface_torque(cells[i]->get_self()) * cells[i]->get_sf_axis(0);
	t.y += cell_surface_torque(cells[i]->get_self()) * cells[i]->get_sf_axis(1);
	t.z += cell_surface_torque(cells[i]->get_self()) * cells[i]->get_sf_axis(2);
	
	if (xy == 1) {
		//Fz = 0;
		t.x = 0;
		t.y = 0;
	}
	if (xz == 1) {
		t.x = 0;
		t.z = 0;
	}
	
	return t;
	/*
	if (dof==0) {
		return tx;
	} else if (dof==1) {
		return ty;
	} else if (dof==2) {
		return tz;
	}
	cout << "NET TORQUE ELON ERROR" << endl;
	return 0;*/
	
}

myVec ext_force_div(int i, int d1) {
	// Returns the force on daughter X of cells[i] during division from
	// 1) other cells
	// To do: surface forces, cell-cell adhesion
	
	// "i" is between 0 and the length of vector "cells"
	// daughter is 1 or 2
	// dof is 0, 1 or 2 (for x, y, or z)
	
	//double Fx, Fy, Fz;
	int d2;
	myVec F, Fi;
	
	F.x = 0;
	F.y = 0;
	F.z = 0;
	
	// Calculate surface force:
	F.z += cell_surface_force(cells[i]->get_daughter(d1));
	
	// Loop over all bodies
  	for (int j = 0; j != cells.size(); j++) {
		
		// Calculate forces due to other bodies:
		if (i != j and cells[j]->get_dividing() == 1) {
			
			Fi = cell_cell_force(2*R, cells[i]->get_daughter(d1), cells[j]->get_daughter(1));
			F.x += Fi.x;
			F.y += Fi.y;
			F.z += Fi.z;
			
			Fi = cell_cell_force(2*R, cells[i]->get_daughter(d1), cells[j]->get_daughter(2));
			F.x += Fi.x;
			F.y += Fi.y;
			F.z += Fi.z;
			
			// Daughter 1 of body j
			//F.x += cell_cell_force(2*R, cells[i]->get_daughter(d1), cells[j]->get_daughter(1));
			//F.y += cell_cell_force(2*R, cells[i]->get_daughter(d1), cells[j]->get_daughter(1));
			//F.z += cell_cell_force(2*R, cells[i]->get_daughter(d1), cells[j]->get_daughter(1));
			
			// Daughter 2 of body j
			//F.x += cell_cell_force(2*R, cells[i]->get_daughter(d1), cells[j]->get_daughter(2));
			//F.y += cell_cell_force(2*R, cells[i]->get_daughter(d1), cells[j]->get_daughter(2));
			//F.z += cell_cell_force(2*R, cells[i]->get_daughter(d1), cells[j]->get_daughter(2));
			
		} else if (i != j and cells[j]->get_dividing() == 0) {
			
			Fi = cell_cell_force(2*R, cells[i]->get_daughter(d1), cells[j]->get_self());
			F.x += Fi.x;
			F.y += Fi.y;
			F.z += Fi.z;
			
			//F.x += cell_cell_force(2*R, cells[i]->get_daughter(d1), cells[j]->get_self());
			//F.y += cell_cell_force(2*R, cells[i]->get_daughter(d1), cells[j]->get_self());
			//F.z += cell_cell_force(2*R, cells[i]->get_daughter(d1), cells[j]->get_self());
			
		}
		
		if (i == j) {
			// Determine other daughter:
			d2 = 1;
			if (d1 == 1) {
				d2 = 2;
			}
			
			
			Cell* daughter_1 = cells[i]->get_daughter(d1);
			Cell* daughter_2 = cells[i]->get_daughter(d2);
			
			daughter_1->set_l(cells[i]->get_l()/2.0 - 2*R + cells[i]->get_d());
			daughter_2->set_l(cells[i]->get_l()/2.0 - 2*R + cells[i]->get_d());
			
			Fi = cell_cell_force(2*R, daughter_1, daughter_2);
			F.x += Fi.x;
			F.y += Fi.y;
			F.z += Fi.z;
			
			// Calculate force due to other daughter
			//F.x += cell_cell_force(2*R, daughter_1, daughter_2);
			//F.y += cell_cell_force(2*R, daughter_1, daughter_2);
			//F.z += cell_cell_force(2*R, daughter_1, daughter_2);
			
		}
		
  	}
	
	if (xy == 1) {
		F.z = 0;
	}
	if (xz == 1) {
		F.y = 0;
	}
	
	return F;
	
	/*
	if (dof==0) {
		return Fx;
	} else if (dof==1) {
		return Fy;
	} else if (dof==2) {
		return Fz;
	}
	cout << "EXT FORCE DIV ERROR" << endl;
	return 0;*/
	
}

myVec ext_torque_div(int i, int d1) {
	// Returns the torque on daughter d1 of cells[i] during division from external forces
	// Currently: cell-cell forces. To do: cell-cell adhesion, surface forces
	
	// "i" is between 0 and the length of vector "cells"
	// daughter is 1 or 2
	// dof is 0, 1 or 2 (for x, y, or z)
	
	double tx, ty, tz;
	int d2;
	
	myVec t, ti;
	
	t.x = 0;
	t.y = 0;
	t.z = 0;
	
	// Surface torque
	t.x += cell_surface_torque(cells[i]->get_daughter(d1)) * cells[i]->get_sf_axis(0);
	t.y += cell_surface_torque(cells[i]->get_daughter(d1)) * cells[i]->get_sf_axis(1);
	t.z += cell_surface_torque(cells[i]->get_daughter(d1)) * cells[i]->get_sf_axis(2);
	
	// Loop over all bodies
  	for (int j = 0; j != cells.size(); j++) {
		
		// Calculate torques due to other bodies:
		if (i != j) {
			
			if (cells[j]->get_dividing() == 1) {
				// Daughter 1 of body j
				
				ti = cell_cell_torque(2*R, cells[i]->get_daughter(d1), cells[j]->get_daughter(1));
				t.x += ti.x;
				t.y += ti.y;
				t.z += ti.z;
				
				//tx += cell_cell_torque(2*R, cells[i]->get_daughter(d1), cells[j]->get_daughter(1), 0);
				//ty += cell_cell_torque(2*R, cells[i]->get_daughter(d1), cells[j]->get_daughter(1), 1);
				//tz += cell_cell_torque(2*R, cells[i]->get_daughter(d1), cells[j]->get_daughter(1), 2);
				
				ti = cell_cell_torque(2*R, cells[i]->get_daughter(d1), cells[j]->get_daughter(2));
				t.x += ti.x;
				t.y += ti.y;
				t.z += ti.z;
				
				// Daughter 2 of body j
				//tx += cell_cell_torque(2*R, cells[i]->get_daughter(d1), cells[j]->get_daughter(2), 0);
				//ty += cell_cell_torque(2*R, cells[i]->get_daughter(d1), cells[j]->get_daughter(2), 1);
				//tz += cell_cell_torque(2*R, cells[i]->get_daughter(d1), cells[j]->get_daughter(2), 2);
			} else {
				
				ti = cell_cell_torque(2*R, cells[i]->get_daughter(d1), cells[j]->get_self());
				t.x += ti.x;
				t.y += ti.y;
				t.z += ti.z;
				
				//tx += cell_cell_torque(2*R, cells[i]->get_daughter(d1), cells[j]->get_self(), 0);
				//ty += cell_cell_torque(2*R, cells[i]->get_daughter(d1), cells[j]->get_self(), 1);
				//tz += cell_cell_torque(2*R, cells[i]->get_daughter(d1), cells[j]->get_self(), 2);
				
			}
		}
		
		if (i == j) {
			
			// Determine other daughter:
			d2 = 1;
			if (d1 == 1) {
				d2 = 2;
			}
			
			Cell* daughter_1 = cells[i]->get_daughter(d1);
			Cell* daughter_2 = cells[i]->get_daughter(d2);
			
			daughter_1->set_l(cells[i]->get_l()/2.0 - 2*R + cells[i]->get_d());
			daughter_2->set_l(cells[i]->get_l()/2.0 - 2*R + cells[i]->get_d());
			
			// Calculate torque due to other daughter
			ti = cell_cell_torque(2*R, daughter_1, daughter_2);
			t.x += ti.x;
			t.y += ti.y;
			t.z += ti.z;
			
			//tx += cell_cell_torque(2*R, daughter_1, daughter_2, 0);
			//ty += cell_cell_torque(2*R, daughter_1, daughter_2, 1);
			//tz += cell_cell_torque(2*R, daughter_1, daughter_2, 2);
		}
		
  	}
	
	if (xy == 1) {
		//Fz = 0;
		t.x = 0;
		t.y = 0;
	}
	if (xz == 1) {
		t.x = 0;
		t.z = 0;
	}
	
	return t;
	/*
	if (dof==0) {
		return t.x;
	} else if (dof==1) {
		return t.y;
	} else if (dof==2) {
		return t.z;
	}
	return 0;*/
	
}

myVec spring_force_div(int i, int d1) {
	//return 0;
	double d = cells[i]->get_d();
	double Lf = cells[i]->get_l()/2.0;
	
	double kappa = cells[i]->get_spring_const();
	double x1 = cells[i]->get_x1();
	double y1 = cells[i]->get_y1();
	double z1 = cells[i]->get_z1();
	double x2 = cells[i]->get_x2();
	double y2 = cells[i]->get_y2();
	double z2 = cells[i]->get_z2();
	double nx1 = cells[i]->get_nx1();
	double ny1 = cells[i]->get_ny1();
	double nz1 = cells[i]->get_nz1();
	double nx2 = cells[i]->get_nx2();
	double ny2 = cells[i]->get_ny2();
	double nz2 = cells[i]->get_nz2();
	
	myVec F;
	F.x = 0;
	F.y = 0;
	F.z = 0;
	
	if (d == 0) {
		return F;
	}
	
	if ( sqrt(pow(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2,2) + 
     pow(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2,2) + 
     pow(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2,2)) == 0) {
		return F;
	}
	
	//cout << "SPRING FORE: " << x2 << " " << nx1 << " " << endl;
	//cout << "kappa: " << kappa << endl;
	//cout << sqrt(pow(((d + L1)*nx1)/2. + ((d + L1)*nx2)/2. - x1 + x2,2) + 
    // pow(((d + L1)*ny1)/2. + ((d + L1)*ny2)/2. - y1 + y2,2) + 
    // pow(((d + L1)*nz1)/2. + ((d + L1)*nz2)/2. - z1 + z2,2))<<endl;
	
	if (d1 == 1) {
		F.x = -(-2*kappa*(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2)*
     (-d + sqrt(pow(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2,2) + 
         pow(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2,2) + 
         pow(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2,2))))/
   sqrt(pow(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2,2) + 
     pow(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2,2) + 
     pow(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2,2));
		F.y = -(-2*kappa*(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2)*
		     (-d + sqrt(pow(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2,2) + 
		         pow(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2,2) + 
		         pow(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2,2))))/
		   sqrt(pow(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2,2) + 
		     pow(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2,2) + 
		     pow(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2,2));
		F.z = -(-2*kappa*(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2)*
     (-d + sqrt(pow(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2,2) + 
         pow(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2,2) + 
         pow(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2,2))))/
   sqrt(pow(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2,2) + 
     pow(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2,2) + 
     pow(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2,2));
		return F;
	} else if (d1 == 2) {
		F.x = -(2*kappa*(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2)*
     (-d + sqrt(pow(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2,2) + 
         pow(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2,2) + 
         pow(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2,2))))/
   sqrt(pow(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2,2) + 
     pow(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2,2) + 
     pow(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2,2));
		F.y = -(2*kappa*(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2)*
     (-d + sqrt(pow(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2,2) + 
         pow(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2,2) + 
         pow(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2,2))))/
   sqrt(pow(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2,2) + 
     pow(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2,2) + 
     pow(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2,2));
		F.z = -(2*kappa*(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2)*
     (-d + sqrt(pow(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2,2) + 
         pow(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2,2) + 
         pow(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2,2))))/
   sqrt(pow(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2,2) + 
     pow(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2,2) + 
     pow(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2,2));
		return F;
	} else {
		cout << "Spring force div error!!!" << endl;
		return F;
	}
	
}

myVec spring_torque_div(int i, int d1) {
	//return 0;
	double d = cells[i]->get_d();
	double Lf = cells[i]->get_l()/2.0;
	
	//cout << Lf << endl;
	
	
	double kappa = cells[i]->get_spring_const();
	double x1 = cells[i]->get_x1();
	double y1 = cells[i]->get_y1();
	double z1 = cells[i]->get_z1();
	double x2 = cells[i]->get_x2();
	double y2 = cells[i]->get_y2();
	double z2 = cells[i]->get_z2();
	double nx1 = cells[i]->get_nx1();
	double ny1 = cells[i]->get_ny1();
	double nz1 = cells[i]->get_nz1();
	double nx2 = cells[i]->get_nx2();
	double ny2 = cells[i]->get_ny2();
	double nz2 = cells[i]->get_nz2();
	
	myVec F;
	F.x = 0;
	F.y = 0;
	F.z = 0;
	
	if (d == 0) {
		return F;
	}
	
	if ( sqrt(pow(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2,2) + 
     pow(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2,2) + 
     pow(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2,2)) == 0) {
		return F;
	}
	
	if (d1 == 1) {
		F.x = -(kappa*(-((d + Lf)*nz1*(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2)) + 
       (d + Lf)*ny1*(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2))*
     (-d + sqrt(pow(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2,2) + 
         pow(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2,2) + 
         pow(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2,2))))/
   sqrt(pow(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2,2) + 
     pow(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2,2) + 
     pow(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2,2));
		F.y = -(kappa*((d + Lf)*nz1*(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2) - 
       (d + Lf)*nx1*(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2))*
     (-d + sqrt(pow(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2,2) + 
         pow(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2,2) + 
         pow(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2,2))))/
   sqrt(pow(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2,2) + 
     pow(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2,2) + 
     pow(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2,2));
		F.z = -(kappa*(-((d + Lf)*ny1*(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2)) + 
       (d + Lf)*nx1*(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2))*
     (-d + sqrt(pow(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2,2) + 
         pow(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2,2) + 
         pow(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2,2))))/
   sqrt(pow(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2,2) + 
     pow(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2,2) + 
     pow(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2,2));
		
		return F;
	} else if (d1 == 2) {
		F.x = -(kappa*(-((d + Lf)*nz2*(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2)) + 
       (d + Lf)*ny2*(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2))*
     (-d + sqrt(pow(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2,2) + 
         pow(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2,2) + 
         pow(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2,2))))/
   sqrt(pow(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2,2) + 
     pow(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2,2) + 
     pow(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2,2));
		F.y = -(kappa*((d + Lf)*nz2*(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2) - 
       (d + Lf)*nx2*(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2))*
     (-d + sqrt(pow(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2,2) + 
         pow(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2,2) + 
         pow(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2,2))))/
   sqrt(pow(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2,2) + 
     pow(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2,2) + 
     pow(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2,2));
		F.z = -(kappa*(-((d + Lf)*ny2*(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2)) + 
       (d + Lf)*nx2*(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2))*
     (-d + sqrt(pow(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2,2) + 
         pow(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2,2) + 
         pow(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2,2))))/
   sqrt(pow(((d + Lf)*nx1)/2. + ((d + Lf)*nx2)/2. - x1 + x2,2) + 
     pow(((d + Lf)*ny1)/2. + ((d + Lf)*ny2)/2. - y1 + y2,2) + 
     pow(((d + Lf)*nz1)/2. + ((d + Lf)*nz2)/2. - z1 + z2,2));
		
		return F;
	} else {
		cout << "Spring force div error!!!" << endl;
		return F;
	}
	
}

myVec net_force_div(int i, int d1) {
	// Returns the net force on cells[i] during division from
	// external forces and spring
	
	// "i" is between 0 and the length of vector "cells"
	// dof is 0, 1 or 2 (for x, y, or z)
	
	myVec F, Fe, Fs;
	Fe = ext_force_div(i, d1);
	Fs = spring_force_div(i, d1);
	
	F.x = Fe.x + Fs.x;
	F.y = Fe.y + Fs.y;
	F.z = Fe.z + Fs.z;
	
	//double Fx = ext_force_div(i, d1, 0) + spring_force_div(i, d1, 0);
	//double Fy = ext_force_div(i, d1, 1) + spring_force_div(i, d1, 1);
	//double Fz = ext_force_div(i, d1, 2) + spring_force_div(i, d1, 2);
	
	//cout <<  ext_force_div(i, d1, 0) << " " << spring_force_div(i, d1, 0) << " " << cells[i]->get_d() << endl;
	
	if (xy == 1) {
		F.z = 0;
	}
	if (xz == 1) {
		F.y = 0;
	}
	
	return F;
	
	/*
	if (dof==0) {
		return Fx;
	} else if (dof==1) {
		return Fy;
	} else if (dof==2) {
		return Fz;
	}
	cout << "NET FORCE DIV ERROR" << endl;
	return 0;*/
}

myVec net_torque_div(int i, int d1) {
	// Returns the net force on cells[i] during division from
	// external forces and spring
	
	// "i" is between 0 and the length of vector "cells"
	// dof is 0, 1 or 2 (for x, y, or z)
	
	myVec F, Fe, Fs;
	Fe = ext_torque_div(i, d1);
	Fs = spring_torque_div(i, d1);
	
	F.x = Fe.x + Fs.x;
	F.y = Fe.y + Fs.y;
	F.z = Fe.z + Fs.z;
	
	if (xy == 1) {
		F.x = 0;
		F.y = 0;
	}
	if (xz == 1) {
		F.x = 0;
		F.z = 0;
	}
	
	return F;
	
	/*
	if (dof==0) {
		return Fx;
	} else if (dof==1) {
		return Fy;
	} else if (dof==2) {
		return Fz;
	}
	cout << "NET TORQUE DIV ERROR" << endl;
	return 0;*/
}

int grow_func (double t, const double y[], double f[], void *params) {
  (void)(t);

  // Calculate the rhs of the differential equation, dy/dt = ?
  // Contributions from: 1) ambient viscosity, 2) cell growth, 3) cell-cell pushing
  // To do: cell-cell sticking, cell-surface pushing, cell-surface adhesion, surface viscosity
  
  // The degrees of freedom are as follows:
  // f[0] = cell 1, x position
  // f[1] = cell 1, y pos
  // f[2] = cell 1, z pos
  // f[3] = cell 1, angle
  // f[4] = cell 1, volume
  // f[5] cell 2, ...
  
  double tx, ty, tz, t_net, nx1, ny1, nz1, nx2, ny2, nz2, l, d, nu1;
  double dx, dy, dz;
  myVec F, T;
  
  // Calculate the force on cell i
  int dof = 0;
  int i = 0;
  //while (i < cells.size()) {  
  while (dof < MAX_DOF) {  
	  if (i >= cells.size()) {
	  	  f[dof] = 0;
		  dof = dof + 1;
	  }
	  else if (cells[i]->get_dividing() == 0) {
		  nx1 = y[dof+3];
		  ny1 = y[dof+4];
		  nz1 = y[dof+5];
		  l = y[dof+6];
		  
		  cells[i]->set_pos(y[dof], y[dof+1], y[dof+2]);
		  cells[i]->set_n1(nx1, ny1, nz1);
		  cells[i]->set_l(l);
		  
		  dx = noise_level * rand()/(static_cast<double>(RAND_MAX));
		  dy = noise_level * rand()/(static_cast<double>(RAND_MAX));
		  dz = noise_level * rand()/(static_cast<double>(RAND_MAX));
		  
		  dx = 0;
		  dy = 0;
		  dz = 0;
		  
	  	  if (xy == 1) {
	  		  dz = 0;
	  	  }
	  	  if (xz == 1) {
	  	  	  dy = 0;
	  	  }
		  
		  nu1 = cell_surface_viscosity(cells[i]->get_self());
		  
		  F = net_force_elon(i);
		  
		  f[dof+0] = (1.0/ (NU_0 + nu1) ) * (1.0/(l)) * (F.x + dx);
		  f[dof+1] = (1.0/ (NU_0 + nu1) ) * (1.0/(l)) * (F.y + dy);
		  f[dof+2] = (1.0/ (NU_0 +  0 ) ) * (1.0/(l)) * (F.z + dz);
	  	  
		  dx = noise_level * rand()/(static_cast<double>(RAND_MAX));
		  dy = noise_level * rand()/(static_cast<double>(RAND_MAX));
		  dz = noise_level * rand()/(static_cast<double>(RAND_MAX));
		  
	  	  if (xy == 1) {
	  		  dx = 0;
			  dy = 0;
	  	  }
	  	  if (xz == 1) {
	  	  	  dx = 0;
			  dz = 0;
	  	  }
		  
		  T = net_torque_elon(i);
		  
	  	  tx = T.x + dx;
		  ty = T.y + dy;
		  tz = T.z + dz;
	      
		  //double farz = 1.0;
		  f[dof+3] = (nz1*ty - ny1*tz)/(pow(l,3)*NU_0*T_0);
		  f[dof+4] = (-(nz1*tx) + nx1*tz)/(pow(l,3)*NU_0*T_0);
		  f[dof+5] = (ny1*tx - nx1*ty)/(pow(l,3)*NU_0*T_0);
		  
		  f[dof+6] = A * (l + (4*R)/3.);
	  	  //f[dof+6] = 0;
		  
		  i++;
		  dof = dof + DOF_GROW;
	  }
	  else if (cells[i]->get_dividing() == 1) {
		  l = cells[i]->get_l()/2.0;
		  nx1 = y[dof+3];
		  ny1 = y[dof+4];
		  nz1 = y[dof+5];
		  nx2 = y[dof+9];
		  ny2 = y[dof+10];
		  nz2 = y[dof+11];
		  d = y[dof+12];
		  
		  //cout << nx2 << " " << ny2 << " " << nz2 << endl;
		  
		  //if (d > 6) {
			//  cout << "SET d ERROR" << endl;
		  //}
		  
		  cells[i]->set_d(d);
		  cells[i]->set_x1(y[dof], y[dof+1], y[dof+2]);
		  cells[i]->set_x2(y[dof+6], y[dof+7], y[dof+8]);
		  cells[i]->set_n1(nx1, ny1, nz1);
		  cells[i]->set_n2(nx2, ny2, nz2);
		  
		  dx = noise_level * rand()/(static_cast<double>(RAND_MAX));
		  dy = noise_level * rand()/(static_cast<double>(RAND_MAX));
		  dz = noise_level * rand()/(static_cast<double>(RAND_MAX));
		  
		  dx = 0;
		  dy = 0;
		  dz = 0;
		  
	  	  if (xy == 1) {
	  		  dz = 0;
	  	  }
	  	  if (xz == 1) {
	  	  	  dy = 0;
	  	  }
		  
		  nu1 = cell_surface_viscosity(cells[i]->get_daughter(1));
		  
		  F = net_force_div(i, 1);
		  
		  f[dof+0] = (1.0/ (NU_0 + nu1) ) * (1.0/(l)) * (F.x + dx);
		  f[dof+1] = (1.0/ (NU_0 + nu1) ) * (1.0/(l)) * (F.y + dy);
		  f[dof+2] = (1.0/ (NU_0 +  0 ) ) * (1.0/(l)) * (F.z + dz);
		  
		  dx = noise_level * rand()/(static_cast<double>(RAND_MAX));
		  dy = noise_level * rand()/(static_cast<double>(RAND_MAX));
		  dz = noise_level * rand()/(static_cast<double>(RAND_MAX));
		  
	  	  if (xy == 1) {
	  		  dx = 0;
			  dy = 0;
	  	  }
	  	  if (xz == 1) {
	  	  	  dx = 0;
			  dz = 0;
	  	  }
		  
		  //tx = net_torque_div(i, 1, 0) + dx;
		  //ty = net_torque_div(i, 1, 1) + dy;
		  //tz = net_torque_div(i, 1, 2) + dz;
		  
		  T = net_torque_div(i, 1);
		  
	  	  tx = T.x + dx;
		  ty = T.y + dy;
		  tz = T.z + dz;
		  
		  f[dof+3] = (nz1*ty - ny1*tz)/(pow(l,3)*NU_0*T_0);
		  f[dof+4] = (-(nz1*tx) + nx1*tz)/(pow(l,3)*NU_0*T_0);
		  f[dof+5] = (ny1*tx - nx1*ty)/(pow(l,3)*NU_0*T_0);
		  
		  dx = noise_level * rand()/(static_cast<double>(RAND_MAX));
		  dy = noise_level * rand()/(static_cast<double>(RAND_MAX));
		  dz = noise_level * rand()/(static_cast<double>(RAND_MAX));
		  
		  dx = 0;
		  dy = 0;
		  dz = 0;
		  
	  	  if (xy == 1) {
	  		  dz = 0;
	  	  }
	  	  if (xz == 1) {
	  	  	  dy = 0;
	  	  }
		  
		  nu1 = cell_surface_viscosity(cells[i]->get_daughter(2));
		  
		  F = net_force_div(i, 2);
		  
		  f[dof+6] = (1.0/ (NU_0 + nu1) ) * (1.0/(l)) * (F.x + dx);
		  f[dof+7] = (1.0/ (NU_0 + nu1) ) * (1.0/(l)) * (F.y + dy);
		  f[dof+8] = (1.0/ (NU_0 +  0 ) ) * (1.0/(l)) * (F.z + dz);
		  
		  dx = noise_level * rand()/(static_cast<double>(RAND_MAX));
		  dy = noise_level * rand()/(static_cast<double>(RAND_MAX));
		  dz = noise_level * rand()/(static_cast<double>(RAND_MAX));
		  
	  	  if (xy == 1) {
	  		  dx = 0;
			  dy = 0;
	  	  }
	  	  if (xz == 1) {
	  	  	  dx = 0;
			  dz = 0;
	  	  }
		  
		  //tx = net_torque_div(i, 2, 0) + dx;
		  //ty = net_torque_div(i, 2, 1) + dy;
		  //tz = net_torque_div(i, 2, 2) + dz;
		  T = net_torque_div(i, 1);
		  
	  	  tx = T.x + dx;
		  ty = T.y + dy;
		  tz = T.z + dz;
		  
		  f[dof+9] = (nz2*ty - ny2*tz)/(pow(l,3)*NU_0*T_0);
		  f[dof+10] = (-(nz2*tx) + nx2*tz)/(pow(l,3)*NU_0*T_0);
		  f[dof+11] = (ny2*tx - nx2*ty)/(pow(l,3)*NU_0*T_0);
		  
		  if (cells[i]->get_d() > 2*R) {
			  cout << "Problem w/ d" << endl;
			  
		  }
		  
		  if (d >= 2*R) {
			  f[dof+12] = 0;
		  } else {
			  f[dof+12] = (A*(pow(cells[i]->get_d(),3) - 12*(cells[i]->get_d() + cells[i]->get_l())*pow(R,2) - 16*pow(R,3)))/
		   (3.*(pow(cells[i]->get_d(),2) - 4*pow(R,2)));
		  }
		  
		  
		  //cout << "force on d: " << (3.*(pow(cells[i]->get_d(),2) - 4*pow(R,2))) << " " << f[dof+12] << endl;
		  
		  i++;
		  dof = dof + DOF_DIV;
	  }
  }
  
  return GSL_SUCCESS;

}

void grow_cells(double tf) {
	// Evolve the system as each cell grows by elongation.
	double y[MAX_DOF];
	
	string my_name = "output/biofilm";
	ofstream myfile;
	string my_mono_name = my_name+"-line.txt";
	myfile.open(my_mono_name, ios::app);
	
	unsigned long dim = MAX_DOF;
	
	const gsl_odeiv2_step_type * TT = gsl_odeiv2_step_rk8pd;

	gsl_odeiv2_step * s = gsl_odeiv2_step_alloc (TT, dim);
	gsl_odeiv2_control * c = gsl_odeiv2_control_y_new (1e-6, 1e-6);
	gsl_odeiv2_evolve * e = gsl_odeiv2_evolve_alloc (dim);

	gsl_odeiv2_system sys = {grow_func, nullptr, dim, nullptr};
	
	int reset;
  	int dof = 0;
	int j = 0;
	
	while (dof < MAX_DOF) {
		if (j >= cells.size()) {
			y[dof] = 0;
			dof = dof + 1;
		} else if (cells[j]->get_dividing() == 0) {
	  		y[dof+0] = cells[j]->get_x();
	  		y[dof+1] = cells[j]->get_y();
	  		y[dof+2] = cells[j]->get_z();
	  		y[dof+3] = cells[j]->get_nx1();
	  		y[dof+4] = cells[j]->get_ny1();
	  		y[dof+5] = cells[j]->get_nz1();
	  		y[dof+6] = cells[j]->get_l();
			
			j++;
	  		dof = dof + DOF_GROW;
		} else if (cells[j]->get_dividing() == 1) {
	  		y[dof+0] = cells[j]->get_x1();
	  		y[dof+1] = cells[j]->get_y1();
	  		y[dof+2] = cells[j]->get_z1();
	  		y[dof+3] = cells[j]->get_nx1();
	  		y[dof+4] = cells[j]->get_ny1();
	  		y[dof+5] = cells[j]->get_nz1();
			y[dof+6] = cells[j]->get_x2();
	  		y[dof+7] = cells[j]->get_y2();
	  		y[dof+8] = cells[j]->get_z2();
	  		y[dof+9] = cells[j]->get_nx2();
	  		y[dof+10] = cells[j]->get_ny2();
	  		y[dof+11] = cells[j]->get_nz2();
	  		y[dof+12] = cells[j]->get_d();
			
			j++;
	  		dof = dof + DOF_DIV;
		} else {
			cout << "Grow cells initial condition ERROR" << endl;
		}
  	}
	
	double t = 0.0, tcurrent = tstep;
	double h = h0;
	
	// Evolve the differential equation
	while (t < tf) {
		
		while (t < tcurrent) {
				
			if (reset == 1) {
				
				gsl_odeiv2_step_reset(s);
				gsl_odeiv2_evolve_reset (e);
				h = h0;
				
				dof = 0;
				j = 0;
				while (dof < MAX_DOF) {
					if (j >= cells.size()) {
						y[dof] = 0;
						dof = dof + 1;
					} else if (cells[j]->get_dividing() == 0) {
				  		y[dof+0] = cells[j]->get_x();
				  		y[dof+1] = cells[j]->get_y();
				  		y[dof+2] = cells[j]->get_z();
				  		y[dof+3] = cells[j]->get_nx1();
				  		y[dof+4] = cells[j]->get_ny1();
				  		y[dof+5] = cells[j]->get_nz1();
				  		y[dof+6] = cells[j]->get_l();
			
						j++;
				  		dof = dof + DOF_GROW;
					} else if (cells[j]->get_dividing() == 1) {
				  		y[dof+0] = cells[j]->get_x1();
				  		y[dof+1] = cells[j]->get_y1();
				  		y[dof+2] = cells[j]->get_z1();
				  		y[dof+3] = cells[j]->get_nx1();
				  		y[dof+4] = cells[j]->get_ny1();
				  		y[dof+5] = cells[j]->get_nz1();
						y[dof+6] = cells[j]->get_x2();
				  		y[dof+7] = cells[j]->get_y2();
				  		y[dof+8] = cells[j]->get_z2();
				  		y[dof+9] = cells[j]->get_nx2();
				  		y[dof+10] = cells[j]->get_ny2();
				  		y[dof+11] = cells[j]->get_nz2();
				  		y[dof+12] = cells[j]->get_d();
						
						j++;
				  		dof = dof + DOF_DIV;
					} else {
						cout << "Time prep ERROR" << endl;
					}
			  	}
   		    }
			
			int status = gsl_odeiv2_evolve_apply (e, c, s,
			                                           &sys, 
			                                           &t, tcurrent,
			                                           &h, y);
			
			//cout << "Time: " << t << endl;
			if (status != GSL_SUCCESS) {
				cout << "WTF" << endl;
				gsl_odeiv2_step_reset(s);
				gsl_odeiv2_evolve_reset (e);
				break;
			}
													
	  		dof = 0;
	  		for (int j = 0; j != cells.size(); j++) {
	  			if (cells[j]->get_dividing() == 0) {
					cells[j]->set_pos(y[dof], y[dof+1], y[dof+2]);
					cells[j]->set_n1(y[dof+3], y[dof+4], y[dof+5]);
					cells[j]->set_l(y[dof+6]);
					
					dof = dof + DOF_GROW;
				} else {
	  	  		    cells[j]->set_d(y[dof+12]);
	  	  		    cells[j]->set_x1(y[dof], y[dof+1], y[dof+2]);
	  	  		    cells[j]->set_x2(y[dof+6], y[dof+7], y[dof+8]);
	  	  		    cells[j]->set_n1(y[dof+3], y[dof+4], y[dof+5]);
	  	  		    cells[j]->set_n2(y[dof+9], y[dof+10], y[dof+11]);
					
	  				dof = dof + DOF_DIV;
	  			}
	  		}
			
			reset = 0;
			j = 0;
			dof = 0;
			
			while (j < cells.size()) {
				if (cells[j]->get_dividing() == 0) {
					if (y[dof+6] > cells[j]->get_Lf()) {
						cells[j]->start_division();
						reset = 1;
					}
					j++;
			  		dof = dof + DOF_GROW;
				} else if (cells[j]->get_dividing() == 1) {
					if (y[dof+12] > 2*R) {
						Cell* c1 = cells[j]->get_daughter(1);
						Cell* c2 = cells[j]->get_daughter(2);
						delete cells[j];
						cells.erase(cells.begin()+j, cells.begin()+j+1);
						cells.insert(cells.begin()+j, new Body(c1->get_x(), c1->get_y(), c1->get_z(), c1->get_nx(), c1->get_ny(), c1->get_nz(), c1->get_l()));
						cells.insert(cells.begin()+j, new Body(c2->get_x(), c2->get_y(), c2->get_z(), c2->get_nx(), c2->get_ny(), c2->get_nz(), c2->get_l()));
						
						reset = 1;
						j = j + 1;
					}
					j = j + 1;
			  		dof = dof + DOF_DIV;
				} else {
					cout << "TIME STEP ERROR" << endl;
					break;
				}
			}
			
			
		}
		 
		int numcell = 0;
		for (int j = 0; j != cells.size(); j++) {
			if (cells[j]->get_dividing() == 0) {
				numcell++;
			} else {
				numcell = numcell + 2;
			}
		}
		
		myfile << t << " " << numcell << " " << tcurrent << endl;
		
		dof = 0;
		for (int j = 0; j != cells.size(); j++) {
			if (cells[j]->get_dividing() == 0) {
				myfile << cells[j]->get_x() << " " << cells[j]->get_y() << " " << cells[j]->get_z() << " "
					<< cells[j]->get_nx1() << " " << cells[j]->get_ny1() << " " << cells[j]->get_nz1() << " "
					<< cells[j]->get_l() << " " << cells[j]->get_d() << endl;
				dof = dof + DOF_GROW;
			 } else {
	 			myfile << cells[j]->get_daughter(1)->get_x() << " " << cells[j]->get_daughter(1)->get_y() << " " << cells[j]->get_daughter(1)->get_z() << " "
	 				<< cells[j]->get_daughter(1)->get_nx() << " " << cells[j]->get_daughter(1)->get_ny() << " " << cells[j]->get_daughter(1)->get_nz() << " "
	 				<< cells[j]->get_daughter(1)->get_l() << " " << cells[j]->get_d() << endl;
	 			myfile << cells[j]->get_daughter(2)->get_x() << " " << cells[j]->get_daughter(2)->get_y() << " " << cells[j]->get_daughter(2)->get_z() << " "
	 					<< cells[j]->get_daughter(2)->get_nx() << " " << cells[j]->get_daughter(2)->get_ny() << " " << cells[j]->get_daughter(2)->get_nz() << " "
	 					<< cells[j]->get_daughter(2)->get_l() << " " << cells[j]->get_d() << endl;
				dof = dof + DOF_DIV;
			}
		}
	
		tcurrent = tcurrent + tstep;
	
    }
	
	gsl_odeiv2_evolve_free (e);
	gsl_odeiv2_control_free (c);
	gsl_odeiv2_step_free (s);
	
}

void simple_grow(double tf) {
	// Evolve the system as each cell grows by elongation.
	double y[MAX_DOF];
	
	string my_name = "output/biofilm";
	ofstream myfile;
	string my_mono_name = my_name+"-line.txt";
	myfile.open(my_mono_name, ios::app);
	
	unsigned long dim = MAX_DOF;
	
	const gsl_odeiv2_step_type * TT = gsl_odeiv2_step_rk8pd;

	  gsl_odeiv2_step * s 
	    = gsl_odeiv2_step_alloc (TT, dim);
	  gsl_odeiv2_control * c 
	    = gsl_odeiv2_control_y_new (1e-6, 0.0);
	  gsl_odeiv2_evolve * e 
	    = gsl_odeiv2_evolve_alloc (dim);

	  gsl_odeiv2_system sys = {grow_func, nullptr, dim, nullptr};
	  
	  
	  //cout << cells.size() << endl;
	  
	//int current_dof = 0;
	double l_m, l_d, dx, dy, dz;
	
	int reset;
  	int dof = 0;
	int j = 0;
	//while (j < cells.size()) {
	while (dof < MAX_DOF) {
		if (j >= cells.size()) {
			y[dof] = 0;
			dof = dof + 1;
		} else if (cells[j]->get_dividing() == 0) {
			//cout << "here" << endl;
	  		y[dof+0] = cells[j]->get_x();
	  		y[dof+1] = cells[j]->get_y();
	  		y[dof+2] = cells[j]->get_z();
	  		y[dof+3] = cells[j]->get_nx1();
	  		y[dof+4] = cells[j]->get_ny1();
	  		y[dof+5] = cells[j]->get_nz1();
	  		y[dof+6] = cells[j]->get_l();
			
			j++;
	  		dof = dof + DOF_GROW;
		} else if (cells[j]->get_dividing() == 1) {
	  		y[dof+0] = cells[j]->get_x1();
	  		y[dof+1] = cells[j]->get_y1();
	  		y[dof+2] = cells[j]->get_z1();
	  		y[dof+3] = cells[j]->get_nx1();
	  		y[dof+4] = cells[j]->get_ny1();
	  		y[dof+5] = cells[j]->get_nz1();
			y[dof+6] = cells[j]->get_x2();
	  		y[dof+7] = cells[j]->get_y2();
	  		y[dof+8] = cells[j]->get_z2();
	  		y[dof+9] = cells[j]->get_nx2();
	  		y[dof+10] = cells[j]->get_ny2();
	  		y[dof+11] = cells[j]->get_nz2();
	  		y[dof+12] = cells[j]->get_d();
			
			//cout << "INITIAL CHOICE OF d: " << cells[j]->get_d() << endl;
			
			j++;
	  		dof = dof + DOF_DIV;
		} else {
			cout << "Time prep ERROR" << endl;
		}
  	}
	
  //cout << "So far so good" << endl;
  
  
  double t = 0.0, tcurrent = tstep;
  double h = 1e-6;
	// Evolve the differential equation
  while (t < tf) {
	  	
	  //h = 1e-6;
	  
		while (t < tcurrent) {
			
   		    if (reset == 1) {
				gsl_odeiv2_evolve_reset (e);
				h = 1e-6;
				
				dof = 0;
				j = 0;
				while (dof < MAX_DOF) {
					if (j >= cells.size()) {
						y[dof] = 0;
						dof = dof + 1;
					} else if (cells[j]->get_dividing() == 0) {
						//cout << "here" << endl;
				  		y[dof+0] = cells[j]->get_x();
				  		y[dof+1] = cells[j]->get_y();
				  		y[dof+2] = cells[j]->get_z();
				  		y[dof+3] = cells[j]->get_nx1();
				  		y[dof+4] = cells[j]->get_ny1();
				  		y[dof+5] = cells[j]->get_nz1();
				  		y[dof+6] = cells[j]->get_l();
			
						j++;
				  		dof = dof + DOF_GROW;
					} else if (cells[j]->get_dividing() == 1) {
				  		y[dof+0] = cells[j]->get_x1();
				  		y[dof+1] = cells[j]->get_y1();
				  		y[dof+2] = cells[j]->get_z1();
				  		y[dof+3] = cells[j]->get_nx1();
				  		y[dof+4] = cells[j]->get_ny1();
				  		y[dof+5] = cells[j]->get_nz1();
						y[dof+6] = cells[j]->get_x2();
				  		y[dof+7] = cells[j]->get_y2();
				  		y[dof+8] = cells[j]->get_z2();
				  		y[dof+9] = cells[j]->get_nx2();
				  		y[dof+10] = cells[j]->get_ny2();
				  		y[dof+11] = cells[j]->get_nz2();
				  		y[dof+12] = cells[j]->get_d();
						
						//cout << "INITIAL CHOICE OF d: " << cells[j]->get_d() << endl;
						
						//cout << "SETTING YDOF: " << cells[j]->get_nx2() << endl;
						j++;
				  		dof = dof + DOF_DIV;
					} else {
						cout << "Time prep ERROR" << endl;
					}
			  	}
   		    }
			//if (h > 1e-6) {
			//	h = 1e-6;
			//}
			//cout << setprecision(16) << t << " " << tcurrent << endl;
			
			int status = gsl_odeiv2_evolve_apply (e, c, s,
			                                           &sys, 
			                                           &t, tcurrent,
			                                           &h, y);
			
	  		 dof = 0;
	  		 for (int j = 0; j != cells.size(); j++) {
	  			 if (cells[j]->get_dividing() == 0) {
	  	  		     cells[j]->set_pos(y[dof], y[dof+1], y[dof+2]);
	  	  		     cells[j]->set_n1(y[dof+3], y[dof+4], y[dof+5]);
	  	  		     cells[j]->set_l(y[dof+6]);
					 
					 //cout << cells[j]->get_l() << endl;
					 
	  				 dof = dof + DOF_GROW;
	  			 } else {
					 
	  	  		    cells[j]->set_d(y[dof+12]);
	  	  		    cells[j]->set_x1(y[dof], y[dof+1], y[dof+2]);
	  	  		    cells[j]->set_x2(y[dof+6], y[dof+7], y[dof+8]);
	  	  		    cells[j]->set_n1(y[dof+3], y[dof+4], y[dof+5]);
	  	  		    cells[j]->set_n2(y[dof+9], y[dof+10], y[dof+11]);
					
					//cout << "d: " << y[dof+12] << endl;
					
					//cout << "Division length: " << cells[j]->get_l() << endl;
	  				dof = dof + DOF_DIV;
	  			 }
	  		 }
			
			reset = 0;
			j = 0;
			dof = 0;
			while (j < cells.size()) {
				if (cells[j]->get_dividing() == 0) {
					// If the cell is greater than Lf, divide!
					
					if (y[dof+6] > cells[j]->get_Lf()) {
			  		    //cells[j]->set_pos(y[dof], y[dof+1], y[dof+2]);
			  		    //cells[j]->set_n1(y[dof+3], y[dof+4], y[dof+5]);
			  		    //cells[j]->set_l(y[dof+6]);
						
						//cout << "Starting to divide!" << endl;
						//cells[j]->start_division();
						//cells[j]->simple_division();
						l_m = y[dof+6];
						l_d = (l_m + 2*R)/2.0 - 2*R;
						
						//cout << l_m + 2*R << " " << l_d * 2.0 + 4*R << endl;
						
						//dx = l_d*y[dof+3]/2;
						//dy = l_d*y[dof+4]/2;
						//dz = l_d*y[dof+5]/2;
						
						dx = ((l_m + 2*R) / 4.0) * y[dof+3];
						dy = ((l_m + 2*R) / 4.0) * y[dof+4];
						dz = ((l_m + 2*R) / 4.0) * y[dof+5];
						
						Cell* c1 = cells[j]->get_daughter(1);
						Cell* c2 = cells[j]->get_daughter(2);
						delete cells[j];
						cells.erase(cells.begin()+j, cells.begin()+j+1);
						cells.insert(cells.begin()+j, new Body(y[dof] + dx, y[dof+1] + dy, y[dof+2] + dz, y[dof+3], y[dof+4], y[dof+5], l_d));
						cells.insert(cells.begin()+j, new Body(y[dof] - dx, y[dof+1] - dy, y[dof+2] - dz, y[dof+3], y[dof+4], y[dof+5], l_d));
						
						//cout << "Length after division: " << cells[j]->get_l() << endl;
						
						reset = 1;
						
						j = j + 1;
						//gsl_odeiv2_evolve_reset (e);
						//h = 1e-6;
						
					}
					j++;
			  		dof = dof + DOF_GROW;
				} else if (cells[j]->get_dividing() == 1) {
					
					//cout << "D: " << t << " " << y[dof+8] << endl;
					if (y[dof+12] > 2*R) {
			  		    //cells[j]->set_d(y[dof+12]);
			  		    //cells[j]->set_x1(y[dof], y[dof+1], y[dof+2]);
			  		    //cells[j]->set_x2(y[dof+6], y[dof+7], y[dof+8]);
			  		    //cells[j]->set_n1(y[dof+3], y[dof+4], y[dof+5]);
			  		    //cells[j]->set_n2(y[dof+9], y[dof+10], y[dof+11]);
						
						Cell* c1 = cells[j]->get_daughter(1);
						Cell* c2 = cells[j]->get_daughter(2);
						delete cells[j];
						cells.erase(cells.begin()+j, cells.begin()+j+1);
						cells.insert(cells.begin()+j, new Body(c1->get_x(), c1->get_y(), c1->get_z(), c1->get_nx(), c1->get_ny(), c1->get_nz(), c1->get_l()));
						cells.insert(cells.begin()+j, new Body(c2->get_x(), c2->get_y(), c2->get_z(), c2->get_nx(), c2->get_ny(), c2->get_nz(), c2->get_l()));
						
						//gsl_odeiv2_evolve_reset (e);
						//h = 1e-6;
						reset = 1;
						
						j = j + 1;
					}
					j = j + 1;
			  		dof = dof + DOF_DIV;
				} else {
					cout << "TIME STEP ERROR" << endl;
					break;
				}
			}
		
			 if (status != GSL_SUCCESS) break;
			
		 }
		 
		 
		 
		 int numcell = 0;
		 for (int j = 0; j != cells.size(); j++) {
			 if (cells[j]->get_dividing() == 0) {
				 numcell++;
			 } else {
				 numcell = numcell + 2;
			 }
		 }
		 
		 myfile << t << " " << numcell << endl;
		 
		 dof = 0;
		 for (int j = 0; j != cells.size(); j++) {
			 if (cells[j]->get_dividing() == 0) {
				 myfile << cells[j]->get_x() << " " << cells[j]->get_y() << " " << cells[j]->get_z() << " "
					 	<< cells[j]->get_nx1() << " " << cells[j]->get_ny1() << " " << cells[j]->get_nz1() << " "
						<< cells[j]->get_l() << endl;
				 dof = dof + DOF_GROW;
			 } else {
	 			myfile << cells[j]->get_daughter(1)->get_x() << " " << cells[j]->get_daughter(1)->get_y() << " " << cells[j]->get_daughter(1)->get_z() << " "
	 					<< cells[j]->get_daughter(1)->get_nx() << " " << cells[j]->get_daughter(1)->get_ny() << " " << cells[j]->get_daughter(1)->get_nz() << " "
	 					<< cells[j]->get_daughter(1)->get_l() << endl;
	 			myfile << cells[j]->get_daughter(2)->get_x() << " " << cells[j]->get_daughter(2)->get_y() << " " << cells[j]->get_daughter(2)->get_z() << " "
	 					<< cells[j]->get_daughter(2)->get_nx() << " " << cells[j]->get_daughter(2)->get_ny() << " " << cells[j]->get_daughter(2)->get_nz() << " "
	 					<< cells[j]->get_daughter(2)->get_l() << endl;
				dof = dof + DOF_DIV;
			 }
		 }
	
		 tcurrent = tcurrent + tstep;
			 
    }
	
	gsl_odeiv2_evolve_free (e);
	  gsl_odeiv2_control_free (c);
	  gsl_odeiv2_step_free (s);
	
}

int main(int argc, char * argv[]) {
	cout << "\n\n\n\n\n\n\n";
	cout << "Program running\n";
	
	int trial = atoi(argv[1]);  //Optional input for seed
	srand ((trial+1)*time(NULL));
	
	// Initializations:
	cells.clear();
	cells.reserve(3500);
	
	// Create first cell:
	
	double ti = 0.0; // initial angle
	double nxi = cos(ti);
	double nzi = sin(ti);
	double nyi = 0;
	
	double init_z = -pow(A_0,0.6666666666666666)/
   (2.*pow(2,0.3333333333333333)*pow(E_0,0.6666666666666666)) + R + nzi*L1/2.0;
	
	if (xy == 1) { nzi = 0; }
	if (xz == 1) { nyi = 0; }
	double nm = sqrt(nxi*nxi + nyi*nyi + nzi*nzi);
	cells.push_back(new Body(0, 0, init_z, nxi/nm, nyi/nm, nzi/nm, L1));
	
	// Output data
	string my_name = "output/biofilm";
	ofstream myfile;
	string my_mono_name = my_name+"-line.txt";
	myfile.open(my_mono_name);
	
	myfile << 0 << " " << cells.size() << endl;
	myfile << 0 << " " << 0 << " " << init_z << " "
			<< nxi << " " << nyi << " " << nzi << " "
			<< L1 << endl;
	
	//grow_cells(T);
	simple_grow(T);
	
	cout << "Done" << endl;
	
	return 0;
}


