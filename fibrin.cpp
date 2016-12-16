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
const double L1 = 1; // initial length of cylinder
const double L2 = 2 * L1;
const double R = 1.0; // spherocylinder radius
const double D2 = 2*L1 + 4*R; // final length of spherocylinder

const double A = 0.05;
const double T1 = log((2*L1 + 4*R/3)/(L1 + 4*R/3))/A; // growth time
const double T2 = log((-1 + 3*pow(R,2) + 3*L1*pow(R,2) + 2*pow(R,3))/(pow(R,2)*(3*L1 + 2*R)))/A; // growth time

const double NU = 1.0; // ambient viscosity
const double E_0 = 100.0; // the elastic modulus of cell body
const double A_0 = 10.0; // surface force

const double DOF_GROW = 5; // degrees of freedom (growing)
const double DOF_DIV = 8; // degrees of freedom (division)

const double tstep = 0.1;
// Additional useful constants
//const double A = log(2.0)/T; // growth rate of spherocylinder
// Note: pi is given by M_PI

int maxgen = 3; // how many generations does the colony grow?
const double xy = 1; // Confine to xy plane?
const double xz = 0; // Confine to xz plane?

double vol_to_l(double vol) {
	return (-4*M_PI*pow(R,3) + 3*vol)/(3.*M_PI*pow(R,2));
}

double vol_to_d(double vol) {
	complex<double> mycomplex (648*L1*pow(M_PI,3)*pow(R,2) + 432*pow(M_PI,3)*pow(R,3) - 324*pow(M_PI,2)*vol, sqrt(abs(-186624*pow(M_PI,6)*pow(R,6) + 
		            pow(648*L1*pow(M_PI,3)*pow(R,2) + 432*pow(M_PI,3)*pow(R,3) - 324*pow(M_PI,2)*vol,2))));
	
	//complex<double> mycomplex2 (648*L1*pow(M_PI,3)*pow(R,2) + 432*pow(M_PI,3)*pow(R,3) - 324*pow(M_PI,2)*vol, sqrt(abs(-186624*pow(M_PI,6)*pow(R,6) + 
	//	            pow(648*L1*pow(M_PI,3)*pow(R,2) + 432*pow(M_PI,3)*pow(R,3) - 324*pow(M_PI,2)*vol,2))));
	
	double myvol = (-6*pow(2,0.3333333333333333)*M_PI*pow(R,2)*cos(arg(mycomplex)/3.))/
		pow(pow(648*L1*pow(M_PI,3)*pow(R,2) + 432*pow(M_PI,3)*pow(R,3) - 324*pow(M_PI,2)*vol + 
        pow(pow(-186624*pow(M_PI,6)*pow(R,6) + 
            pow(648*L1*pow(M_PI,3)*pow(R,2) + 432*pow(M_PI,3)*pow(R,3) - 324*pow(M_PI,2)*vol,2),2),0.25)*
         cos(arg(-186624*pow(M_PI,6)*pow(R,6) + 
             pow(648*L1*pow(M_PI,3)*pow(R,2) + 432*pow(M_PI,3)*pow(R,3) - 324*pow(M_PI,2)*vol,2))/2.),2) + 
      sqrt(pow(-186624*pow(M_PI,6)*pow(R,6) + 
          pow(648*L1*pow(M_PI,3)*pow(R,2) + 432*pow(M_PI,3)*pow(R,3) - 324*pow(M_PI,2)*vol,2),2))*
       pow(sin(arg(-186624*pow(M_PI,6)*pow(R,6) + 
            pow(648*L1*pow(M_PI,3)*pow(R,2) + 432*pow(M_PI,3)*pow(R,3) - 324*pow(M_PI,2)*vol,2))/2.),2),
     0.16666666666666666) - (cos(arg(mycomplex)/3.)*
      pow(pow(648*L1*pow(M_PI,3)*pow(R,2) + 432*pow(M_PI,3)*pow(R,3) - 324*pow(M_PI,2)*vol + 
          pow(pow(-186624*pow(M_PI,6)*pow(R,6) + 
              pow(648*L1*pow(M_PI,3)*pow(R,2) + 432*pow(M_PI,3)*pow(R,3) - 324*pow(M_PI,2)*vol,2),2),0.25)*
           cos(arg(-186624*pow(M_PI,6)*pow(R,6) + 
               pow(648*L1*pow(M_PI,3)*pow(R,2) + 432*pow(M_PI,3)*pow(R,3) - 324*pow(M_PI,2)*vol,2))/2.),2) + 
        sqrt(pow(-186624*pow(M_PI,6)*pow(R,6) + 
            pow(648*L1*pow(M_PI,3)*pow(R,2) + 432*pow(M_PI,3)*pow(R,3) - 324*pow(M_PI,2)*vol,2),2))*
         pow(sin(arg(-186624*pow(M_PI,6)*pow(R,6) + 
              pow(648*L1*pow(M_PI,3)*pow(R,2) + 432*pow(M_PI,3)*pow(R,3) - 324*pow(M_PI,2)*vol,2))/2.),2),
       0.16666666666666666))/(6.*pow(2,0.3333333333333333)*M_PI) + 
   (6*pow(2,0.3333333333333333)*sqrt(3)*M_PI*pow(R,2)*
      sin(arg(mycomplex)/3.))/
    pow(pow(648*L1*pow(M_PI,3)*pow(R,2) + 432*pow(M_PI,3)*pow(R,3) - 324*pow(M_PI,2)*vol + 
        pow(pow(-186624*pow(M_PI,6)*pow(R,6) + 
            pow(648*L1*pow(M_PI,3)*pow(R,2) + 432*pow(M_PI,3)*pow(R,3) - 324*pow(M_PI,2)*vol,2),2),0.25)*
         cos(arg(-186624*pow(M_PI,6)*pow(R,6) + 
             pow(648*L1*pow(M_PI,3)*pow(R,2) + 432*pow(M_PI,3)*pow(R,3) - 324*pow(M_PI,2)*vol,2))/2.),2) + 
      sqrt(pow(-186624*pow(M_PI,6)*pow(R,6) + 
          pow(648*L1*pow(M_PI,3)*pow(R,2) + 432*pow(M_PI,3)*pow(R,3) - 324*pow(M_PI,2)*vol,2),2))*
       pow(sin(arg(-186624*pow(M_PI,6)*pow(R,6) + 
            pow(648*L1*pow(M_PI,3)*pow(R,2) + 432*pow(M_PI,3)*pow(R,3) - 324*pow(M_PI,2)*vol,2))/2.),2),
     0.16666666666666666) + (pow(pow(648*L1*pow(M_PI,3)*pow(R,2) + 432*pow(M_PI,3)*pow(R,3) - 
          324*pow(M_PI,2)*vol + pow(pow(-186624*pow(M_PI,6)*pow(R,6) + 
              pow(648*L1*pow(M_PI,3)*pow(R,2) + 432*pow(M_PI,3)*pow(R,3) - 324*pow(M_PI,2)*vol,2),2),0.25)*
           cos(arg(-186624*pow(M_PI,6)*pow(R,6) + 
               pow(648*L1*pow(M_PI,3)*pow(R,2) + 432*pow(M_PI,3)*pow(R,3) - 324*pow(M_PI,2)*vol,2))/2.),2) + 
        sqrt(pow(-186624*pow(M_PI,6)*pow(R,6) + 
            pow(648*L1*pow(M_PI,3)*pow(R,2) + 432*pow(M_PI,3)*pow(R,3) - 324*pow(M_PI,2)*vol,2),2))*
         pow(sin(arg(-186624*pow(M_PI,6)*pow(R,6) + 
              pow(648*L1*pow(M_PI,3)*pow(R,2) + 432*pow(M_PI,3)*pow(R,3) - 324*pow(M_PI,2)*vol,2))/2.),2),
       0.16666666666666666)*sin(arg(mycomplex)/3.))/
    (2.*pow(2,0.3333333333333333)*sqrt(3)*M_PI);
	   
	   if (myvol < 0) {
		   return 0;
	   } else if (vol > 8.0*M_PI*R*R*R/3.0 + 2*M_PI*R*R*L1 - M_PI*(4*R+2*R)*(2*R-2*R)*(2*R-2*R)/12.0) {
		   //cout << "OVER 2" << endl;
		   return 2;
	   } else {
		   return myvol;
	   }
	/*
	
	complex<double> mycomplex (16*M_PI - 3*vol, sqrt(abs(240*pow(M_PI,2) - 96*M_PI*vol + 9*pow(vol,2))));
	
	return ((2*pow(2,0.3333333333333333)*pow(M_PI,0.6666666666666666) + 
       pow(pow(16*M_PI - 3*vol,2) + 3*sqrt(pow(80*pow(M_PI,2) - 32*M_PI*vol + 3*pow(vol,2),2)) + 
         2*sqrt(3)*(16*M_PI - 3*vol)*pow(pow(80*pow(M_PI,2) - 32*M_PI*vol + 3*pow(vol,2),2),0.25)*
          cos(arg(80*pow(M_PI,2) - 32*M_PI*vol + 3*pow(vol,2))/2.),0.3333333333333333))*
     (-cos(arg(mycomplex)/3.) + 
       sqrt(3)*sin(arg(mycomplex)/3.)))/(pow(2,0.6666666666666666)*pow(M_PI,0.3333333333333333)*
     pow(pow(16*M_PI - 3*vol,2) + 3*sqrt(pow(80*pow(M_PI,2) - 32*M_PI*vol + 3*pow(vol,2),2)) + 
       2*sqrt(3)*(16*M_PI - 3*vol)*pow(pow(80*pow(M_PI,2) - 32*M_PI*vol + 3*pow(vol,2),2),0.25)*
        cos(arg(80*pow(M_PI,2) - 32*M_PI*vol + 3*pow(vol,2))/2.),0.16666666666666666));*/
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

class Body {
	// One membrane-enclosed entity. When d = 0, contains a single cell.
	// When d > 0, contains two cells
	private:
		double x, y, z; 	// Center of mass of the cell
		double nx1, ny1, nz1, nx2, ny2, nz2, l, d;
	public:
		Body(double, double, double, double, double, double);
		void set_pos(double mx, double my, double mz) {
			x = mx;
			y = my;
			z = mz;
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
		double get_nx1() { return nx1; }
		double get_ny1() { return ny1; }
		double get_nz1() { return nz1; }
		double get_nx2() { return nx2; }
		double get_ny2() { return ny2; }
		double get_nz2() { return nz2; }
		double get_l() { return l; }
		double get_d() { return d; }
		double get_vol() {
			//cout << l*M_PI*pow(R,2) + (4*M_PI*pow(R,3))/3. << " " << 8.0*M_PI*R*R*R/3.0 + 2*M_PI*R*R*L1 - M_PI*(4*R+d)*(2*R-d)*(2*R-d)/12.0 << endl;
			if (d == 0) {
				return l*M_PI*pow(R,2) + (4*M_PI*pow(R,3))/3.;
			} else {
				return 8.0*M_PI*R*R*R/3.0 + 2*M_PI*R*R*L1 - M_PI*(4*R+d)*(2*R-d)*(2*R-d)/12.0;
			}
		}
		double get_theta_axis(int dof) {
			double t0, t1, t2, tn;
			t0 = cross(nx1, ny1, nz1, nx2, ny2, nz2, 0);
			t1 = cross(nx1, ny1, nz1, nx2, ny2, nz2, 1);
			t2 = cross(nx1, ny1, nz1, nx2, ny2, nz2, 2);
			
			if (t0 == 0 and t1 == 0 and t2 == 0) {
				t0 = cross(nx1+1, ny1, nz1, nx2, ny2, nz2, 0);
				t1 = cross(nx1+1, ny1, nz1, nx2, ny2, nz2, 1);
				t2 = cross(nx1+1, ny1, nz1, nx2, ny2, nz2, 2);
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
		double get_internal_torque() {
			//double q = d;
			//if (q < 0.01) {
			//	q = 0.000001;
			//}
			double kappa = (2*R - d) / (2*R*d);
			if (kappa > 500) {
				kappa = 500;
			}
			return kappa*(M_PI - acos(nx1*nx2 + ny1*ny2 + nz1*nz2));
		}
		Cell* get_self() {
			Cell* mother = new Cell(x, y, z, this->get_nx1(), this->get_ny1(), this->get_nz1());
			mother->set_l(l);
			return mother;
		}
		Cell* get_daughter(int c) {
			if (c == 1) {
				double x1 = x + (d * nx1 + L1 * nx1 * 2.0 - d * nx2)/4.0;
				double y1 = y + (d * ny1 + L1 * ny1 * 2.0 - d * ny2)/4.0;
				double z1 = z + (d * nz1 + L1 * nz1 * 2.0 - d * nz2)/4.0;
				return new Cell(x1, y1, z1, this->get_nx1(), this->get_ny1(), this->get_nz1());
			}
			if (c == 2) {
				double x2 = x + (d * nx2 + L1 * nx2 * 2.0 - d * nx1)/4.0;
				double y2 = y + (d * ny2 + L1 * ny2 * 2.0 - d * ny1)/4.0;
				double z2 = z + (d * nz2 + L1 * nz2 * 2.0 - d * nz1)/4.0;
				return new Cell(x2, y2, z2, this->get_nx2(), this->get_ny2(), this->get_nz2());
			} else {
				cout << "RETRIEVE DAUGHTER ERROR" << endl;
			}
		}
};

Body::Body(double mx, double my, double mz, double mnx, double mny, double mnz) {
	x = mx;
	y = my;
	z = mz;
	nx1 = mnx;
	ny1 = mny;
	nz1 = mnz;
	
	nx2 = -nx1;
	ny2 = -ny2;
	nz2 = -nz2;
	
	l = L1;
	d = 0;
}

vector<Body *> cells;

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

double cell_cell_force(Cell* cell_1, Cell* cell_2, int dof) {
	// Return the force acting on cell 1 caused by cell 2
	// dof = 0 (x coord), dof = 1 (y), dof = 2 (z)
	
	double rx, ry, rz;
	
	double Fji, vx, vy, vz, wx, wy, wz, sx, sy, sz, sm, tx, ty, tz, t_net, overlap;
	
	double Fx, Fy, Fz;
	
	overlap = 2*R - get_overlap(cell_1, cell_2);
	
  	if (overlap > 0) {
		  //contact on cell i
		  vx = get_overlap_vec(cell_1, cell_2, 0);
		  vy = get_overlap_vec(cell_1, cell_2, 1);
		  vz = get_overlap_vec(cell_1, cell_2, 2);

		  //contact on cell j
		  wx = get_overlap_vec(cell_1, cell_2, 3);
		  wy = get_overlap_vec(cell_1, cell_2, 4);
		  wz = get_overlap_vec(cell_1, cell_2, 5);

		  sx = vx - wx;
		  sy = vy - wy;
		  sz = vz - wz;
		  sm = sqrt(sx*sx + sy*sy + sz*sz);
		  
		  rx = vx - cell_1->get_x();
		  ry = vy - cell_1->get_y();
		  rz = vz - cell_1->get_z();

		  if (sm == 0) {
			  cout << "SM ERROR!" << endl;
			  sx = cell_1->get_x() - cell_2->get_x();
			  sy = cell_1->get_y() - cell_2->get_y();
			  sz = cell_1->get_z() - cell_2->get_z();
			  sm = sqrt(sx*sx + sy*sy + sz*sz);
		  }
  		  
		  Fji = E_0 * pow(overlap, 3.0/2.0);
		  
		  // Calculate force
		  Fx = Fji*sx/sm;
		  Fy = Fji*sy/sm;
		  Fz = Fji*sz/sm;
		
	}
	else {
		Fx = 0; Fy = 0; Fz = 0;
	}
	
	if (dof==0) {
		return Fx;
	} else if (dof==1) {
		return Fy;
	} else if (dof==2) {
		return Fz;
	}
}

double cell_cell_torque(Cell* cell_1, Cell* cell_2, int dof) {
	// Return the torque acting on cell 1 caused by cell 2
	// dof = 0 (x coord), dof = 1 (y), dof = 2 (z)
	
	double rx, ry, rz;
	
	double Fji, vx, vy, vz, wx, wy, wz, sx, sy, sz, sm, tx, ty, tz, t_net, overlap;
	
	overlap = 2*R - get_overlap(cell_1, cell_2);
	
  	if (overlap > 0) {
		  //contact on cell i
		  vx = get_overlap_vec(cell_1, cell_2, 0);
		  vy = get_overlap_vec(cell_1, cell_2, 1);
		  vz = get_overlap_vec(cell_1, cell_2, 2);

		  //contact on cell j
		  wx = get_overlap_vec(cell_1, cell_2, 3);
		  wy = get_overlap_vec(cell_1, cell_2, 4);
		  wz = get_overlap_vec(cell_1, cell_2, 5);

		  sx = vx - wx;
		  sy = vy - wy;
		  sz = vz - wz;
		  sm = sqrt(sx*sx + sy*sy + sz*sz);

		  rx = vx - cell_1->get_x();
		  ry = vy - cell_1->get_y();
		  rz = vz - cell_1->get_z();

		  if (sm == 0) {
			  //cout << "TORQUE ERROR!" << endl;
			  tx = 0;
			  ty = 0;
			  tz = 0;
		  } else {
			  Fji = E_0 * pow(overlap, 3.0/2.0);
	  
			  // Calculate torque
			  tx = cross(rx, ry, rz, Fji*sx/sm, Fji*sy/sm, Fji*sz/sm, 0);
			  ty = cross(rx, ry, rz, Fji*sx/sm, Fji*sy/sm, Fji*sz/sm, 1);
			  tz = cross(rx, ry, rz, Fji*sx/sm, Fji*sy/sm, Fji*sz/sm, 2);
		  }
		  
		  
		
	}
	else {
		tx = 0; ty = 0; tz = 0;
	}
	
	if (dof==0) {
		return tx;
	} else if (dof==1) {
		return ty;
	} else if (dof==2) {
		return tz;
	}
}

double net_force(int i, int d1, int dof) {
	// Returns the net force on daughter X of cells[i]
	
	// "i" is between 0 and the length of vector "cells"
	// daughter is 1 or 2
	// dof is 0, 1 or 2 (for x, y, or z)
	
	double rx, ry, rz;
	
	double Fji, vx, vy, vz, wx, wy, wz, sx, sy, sz, sm;
	
	double Fx, Fy, Fz;
	int d2;
	
	double tx, ty, tz, t_net, overlap;
	
	Fx = 0;
	Fy = 0;
	Fz = 0;
	
	// Loop over all bodies
  	for (int j = 0; j != cells.size(); j++) {
		
		// Calculate forces due to other bodies:
		if (i != j) {
			
			if (cells[i]->get_d() == 0) {
				// cells are growing.
				Fx += cell_cell_force(cells[i]->get_self(), cells[j]->get_self(), 0);
				Fy += cell_cell_force(cells[i]->get_self(), cells[j]->get_self(), 1);
				Fz += cell_cell_force(cells[i]->get_self(), cells[j]->get_self(), 2);
				
			} else {
				// cells are dividing.
				
				// Daughter 1 of body j
				Fx += cell_cell_force(cells[i]->get_daughter(d1), cells[j]->get_daughter(1), 0);
				Fy += cell_cell_force(cells[i]->get_daughter(d1), cells[j]->get_daughter(1), 1);
				Fz += cell_cell_force(cells[i]->get_daughter(d1), cells[j]->get_daughter(1), 2);
				
				// Daughter 2 of body j
				Fx += cell_cell_force(cells[i]->get_daughter(d1), cells[j]->get_daughter(2), 0);
				Fy += cell_cell_force(cells[i]->get_daughter(d1), cells[j]->get_daughter(2), 1);
				Fz += cell_cell_force(cells[i]->get_daughter(d1), cells[j]->get_daughter(2), 2);
				
			}
		}
		
		if (i == j and cells[i]->get_d() > 0) {
			// Determine other daughter:
			d2 = 1;
			if (d1 == 1) {
				d2 = 2;
			}
			
			// Calculate force due to other daughter
			Fx += cell_cell_force(cells[i]->get_daughter(d1), cells[i]->get_daughter(d2), 0);
			Fy += cell_cell_force(cells[i]->get_daughter(d1), cells[i]->get_daughter(d2), 1);
			Fz += cell_cell_force(cells[i]->get_daughter(d1), cells[i]->get_daughter(d2), 2);
		}
		
  	}
	
	if (xy == 1) {
		Fz = 0;
	}
	if (xz == 1) {
		Fy = 0;
	}
	
	if (dof==0) {
		return Fx;
	} else if (dof==1) {
		return Fy;
	} else if (dof==2) {
		return Fz;
	}
	return 0;
	
}

double net_torque(int i, int d1, int dof) {
	// Returns the net torque on daughter X of cells[i]
	
	// "i" is between 0 and the length of vector "cells"
	// daughter is 1 or 2
	// dof is 0, 1 or 2 (for x, y, or z)
	
	double rx, ry, rz;
	
	double Fji, vx, vy, vz, wx, wy, wz, sx, sy, sz, sm;
	
	double Fx, Fy, Fz;
	int d2;
	
	double tx, ty, tz, t_net, overlap;
	
	Fx = 0;
	Fy = 0;
	Fz = 0;
	
	// Loop over all bodies
  	for (int j = 0; j != cells.size(); j++) {
		
		// Calculate torques due to other bodies:
		if (i != j) {
			
			if (d1 == 0) {
				// cells are growing.
				Fx += cell_cell_torque(cells[i]->get_self(), cells[j]->get_self(), 0);
				Fy += cell_cell_torque(cells[i]->get_self(), cells[j]->get_self(), 1);
				Fz += cell_cell_torque(cells[i]->get_self(), cells[j]->get_self(), 2);
				
			} else {
				// cells are dividing.
				
				// Daughter 1 of body j
				Fx += cell_cell_torque(cells[i]->get_daughter(d1), cells[j]->get_daughter(1), 0);
				Fy += cell_cell_torque(cells[i]->get_daughter(d1), cells[j]->get_daughter(1), 1);
				Fz += cell_cell_torque(cells[i]->get_daughter(d1), cells[j]->get_daughter(1), 2);
				
				// Daughter 2 of body j
				Fx += cell_cell_torque(cells[i]->get_daughter(d1), cells[j]->get_daughter(2), 0);
				Fy += cell_cell_torque(cells[i]->get_daughter(d1), cells[j]->get_daughter(2), 1);
				Fz += cell_cell_torque(cells[i]->get_daughter(d1), cells[j]->get_daughter(2), 2);
				
			}
		}
		
		if (i == j and d1 > 0) {
			// Determine other daughter:
			d2 = 1;
			if (d1 == 1) {
				d2 = 2;
			}
			
			// Calculate torque due to other daughter
			Fx += cell_cell_torque(cells[i]->get_daughter(d1), cells[i]->get_daughter(d2), 0);
			Fy += cell_cell_torque(cells[i]->get_daughter(d1), cells[i]->get_daughter(d2), 1);
			Fz += cell_cell_torque(cells[i]->get_daughter(d1), cells[i]->get_daughter(d2), 2);
		}
		
  	}
	
	if (xy == 1) {
		//Fz = 0;
		Fx = 0;
		Fy = 0;
	}
	if (xz == 1) {
		Fx = 0;
		Fz = 0;
	}
	
	if (dof==0) {
		return Fx;
	} else if (dof==1) {
		return Fy;
	} else if (dof==2) {
		return Fz;
	}
	return 0;
	
}

int grow_func (double t, const double y[], double f[], void *params) {
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
	  
	  f[dof] = (1.0/NU) * (1.0/(cells[i]->get_l())) * net_force(i, 1, 0);
	  f[dof+1] = (1.0/NU) * (1.0/(cells[i]->get_l())) * net_force(i, 1, 1);
	  f[dof+2] = (1.0/NU) * (1.0/(cells[i]->get_l())) * net_force(i, 1, 2);
	  
	  f[dof+0] += rand()/(100000*static_cast<double>(RAND_MAX));
	  f[dof+1] += rand()/(100000*static_cast<double>(RAND_MAX));
	  f[dof+2] += rand()/(100000*static_cast<double>(RAND_MAX));
	  
	  if (xy == 1) {
		 f[dof+2] = 0; //confined to xy plane
	  }
	  if (xz == 1) {
		 f[dof+1] = 0; //confined to xz plane
	  }
	  
	  tx = net_torque(i, 0, 0);
	  ty = net_torque(i, 0, 1);
	  tz = net_torque(i, 0, 2);
	  t_net = sqrt(tx*tx + ty*ty + tz*tz);
	  
	  f[dof+3] = (1.0/NU) * pow(1.0/(cells[i]->get_l()), 3) * t_net;
	  
	  f[dof+4] = A * (cells[i]->get_vol());
	  
	  dof = dof + DOF_GROW;
  }
  
  return GSL_SUCCESS;

}

int div_func (double t, const double y[], double f[], void *params) {
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
  
  double tx, ty, tz, t_net, t_int;
  
  int dof = 0;
  
  // Calculate the force on cell i:
  for (int i = 0; i != cells.size(); i++) {
	  
	  f[dof] = (1.0/NU) * (1.0/(cells[i]->get_l())) * net_force(i, 1, 0);
	  f[dof+1] = (1.0/NU) * (1.0/(cells[i]->get_l())) * net_force(i, 1, 1);
	  f[dof+2] = (1.0/NU) * (1.0/(cells[i]->get_l())) * net_force(i, 1, 2);
	  
	  f[dof] += (1.0/NU) * (1.0/(cells[i]->get_l())) * net_force(i, 2, 0);
	  f[dof+1] += (1.0/NU) * (1.0/(cells[i]->get_l())) * net_force(i, 2, 1);
	  f[dof+2] += (1.0/NU) * (1.0/(cells[i]->get_l())) * net_force(i, 2, 2);
	  
	  f[dof+0] += rand()/(100000*static_cast<double>(RAND_MAX));
	  f[dof+1] += rand()/(100000*static_cast<double>(RAND_MAX));
	  f[dof+2] += rand()/(100000*static_cast<double>(RAND_MAX));
	  
	  if (xy == 1) {
		 f[dof+2] = 0; //confined to xy plane
	  }
	  if (xz == 1) {
		 f[dof+1] = 0; //confined to xz plane
	  }
	  
	  // Theta axis 1
	  tx = net_torque(i, 1, 0) * cells[i]->get_theta_axis(0);
	  ty = net_torque(i, 1, 1) * cells[i]->get_theta_axis(1);
	  tz = net_torque(i, 1, 2) * cells[i]->get_theta_axis(2);
	  t_int = cells[i]->get_internal_torque();
	  t_net = -(sqrt(tx*tx + ty*ty + tz*tz) + t_int);
	  //cout << "TNET: " << t_net << " " << t_int << " " << tx << " " << ty << " " << tz << endl;
	  
	  f[dof+3] = (1.0/NU) * pow(1.0/L1, 3) * t_net;
	  
	  // Theta axis 2
	  tx = net_torque(i, 2, 0) * cells[i]->get_theta_axis(0);
	  ty = net_torque(i, 2, 1) * cells[i]->get_theta_axis(1);
	  tz = net_torque(i, 2, 2) * cells[i]->get_theta_axis(2);
	  t_int = cells[i]->get_internal_torque();
	  t_net = sqrt(tx*tx + ty*ty + tz*tz) - t_int;
	  //cout << "TNET: " << t_net << " " << t_int << " " << tx << " " << ty << " " << tz << endl;
	  
	  f[dof+4] = (1.0/NU) * pow(1.0/L1, 3) * t_net;
	  
	  
	  /*
	  // n1 axis
	  tx = net_torque(i, 1, 0) * cells[i]->get_n1_axis(0);
	  ty = net_torque(i, 1, 1) * cells[i]->get_n1_axis(1);
	  tz = net_torque(i, 1, 2) * cells[i]->get_n1_axis(2);
	  t_net = sqrt(tx*tx + ty*ty + tz*tz);
	  f[dof+5] = (1.0/NU) * pow(1.0/L1, 3) * t_net;
	  
	  // n2 axis
	  tx = net_torque(i, 2, 0) * cells[i]->get_n2_axis(0);
	  ty = net_torque(i, 2, 1) * cells[i]->get_n2_axis(1);
	  tz = net_torque(i, 2, 2) * cells[i]->get_n2_axis(2);
	  t_net = sqrt(tx*tx + ty*ty + tz*tz);
	  f[dof+6] = (1.0/NU) * pow(1.0/L1, 3) * t_net;
	  
	  if (xy == 1) {
 		 f[dof+5] = 0; //confined to xy plane
		 f[dof+6] = 0; //confined to xy plane
	  }
	  //if (xz == 1) {
		// f[dof+1] = 0; //confined to xz plane
	  //}
	  */
	  //f[dof+3] = 0;
	  //f[dof+4] = 0;
	  f[dof+5] = 0;
	  f[dof+6] = 0;
	  
	  f[dof+7] = A * (cells[i]->get_vol());
	  //f[dof+7] = A * (y[dof+7]);
	  
	  //cout << "REPORTING THE VOLUME: " << cells[i]->get_vol() << endl;;
	  
	  dof = dof + DOF_DIV;
  }
  
  return GSL_SUCCESS;

}

void grow_cells(int gen) {
	// Evolve the system as each cell grows by elongation.
	double y[5000];
	int dof;
	//int Ni;
	
	string my_name = "output/biofilm";
	ofstream myfile;
	string my_mono_name = my_name+"-line.txt";
	myfile.open(my_mono_name, ios::app);
	
	unsigned long dim = DOF_GROW * cells.size();
	
	const gsl_odeiv2_step_type * TT = gsl_odeiv2_step_rk8pd;

	  gsl_odeiv2_step * s 
	    = gsl_odeiv2_step_alloc (TT, dim);
	  gsl_odeiv2_control * c 
	    = gsl_odeiv2_control_y_new (1e-6, 0.0);
	  gsl_odeiv2_evolve * e 
	    = gsl_odeiv2_evolve_alloc (dim);

	  gsl_odeiv2_system sys = {grow_func, nullptr, dim, nullptr};
	  
  	dof = 0;
  	for (int j = 0; j != cells.size(); j++) {
  		y[dof+0] = cells[j]->get_x();
  		y[dof+1] = cells[j]->get_y();
  		y[dof+2] = cells[j]->get_z();
		
		//if (xy == 1) {
		//	y[dof+2] = 0;
		//}
		
  		y[dof+3] = 0; // "initial condition" for torque is always zero
  		y[dof+4] = cells[j]->get_vol();
		
  		dof = dof + DOF_GROW;
  	}
  
  double t = 0.0, tcurrent = tstep;
  double h = 1e-6;
	// Evolve the differential equation
    while (t < T1) {
		
		while (t < tcurrent) {
			
		  	dof = 0;
		  	for (int j = 0; j != cells.size(); j++) {
		  		y[dof+3] = 0; // "initial condition" for torque is always zero
		  		dof = dof + DOF_GROW;
		  	}
			
			int status = gsl_odeiv2_evolve_apply (e, c, s,
			                                           &sys, 
			                                           &t, tcurrent,
			                                           &h, y);

   			// Output the progress
   			//myfile << gen*T + t << " " << cells.size() << endl;
			
			dof = 0;
			for (int j = 0; j != cells.size(); j++) {
				// Rotate the vector n around t by y[dof+3]:
				// First, obtain t by summing over other cells:
				double tx, ty, tz, t_net, nxi, nyi, nzi, nm;
				
				nxi = cells[j]->get_nx1();
				nyi = cells[j]->get_ny1();
				nzi = cells[j]->get_nz1();
				
				tx = net_torque(j, 1, 0);
				ty = net_torque(j, 1, 1);
				tz = net_torque(j, 1, 2);
				
		  	 	t_net = sqrt(tx*tx + ty*ty + tz*tz);

				cells[j]->set_pos(y[dof+0],y[dof+1],y[dof+2]);

				if (t_net > 0) {
					cells[j]->set_n1(
									rot(nxi, nyi, nzi, tx/t_net, ty/t_net, tz/t_net, y[dof+3], 0),
									rot(nxi, nyi, nzi, tx/t_net, ty/t_net, tz/t_net, y[dof+3], 1),
									rot(nxi, nyi, nzi, tx/t_net, ty/t_net, tz/t_net, y[dof+3], 2)
								   );
				}

				nxi = cells[j]->get_nx1() + rand()/(100000*static_cast<double>(RAND_MAX));
				nyi = cells[j]->get_ny1() + rand()/(100000*static_cast<double>(RAND_MAX));
				nzi = cells[j]->get_nz1() + rand()/(100000*static_cast<double>(RAND_MAX));
				
				if (xy == 1) {
					nzi = 0;
				}
				if (xz == 1) {
					nyi = 0;
				}
				
				nm = sqrt(nxi*nxi + nyi*nyi + nzi*nzi);
				
				// Normalize the orientation vector, to be sure...
				cells[j]->set_n1(nxi/nm, nyi/nm, nzi/nm);
				
				cells[j]->set_l(vol_to_l(y[dof+4]));

				dof = dof + DOF_GROW;
			}
		
			 if (status != GSL_SUCCESS) break;
			
		 }
		 
		myfile << gen*(T1+T2) + t << " " << cells.size() << endl;
		dof = 0;
		for (int j = 0; j != cells.size(); j++) {
			myfile << y[dof+0] << " " << y[dof+1] << " " << y[dof+2] << " "
					<< cells[j]->get_nx1() << " " << cells[j]->get_ny1() << " " << cells[j]->get_nz1() << " "
					<< vol_to_l(y[dof+4]) << endl;
			dof = dof + DOF_GROW;
		}
	
		 tcurrent = tcurrent + tstep;
			 
    }
	
	gsl_odeiv2_evolve_free (e);
	  gsl_odeiv2_control_free (c);
	  gsl_odeiv2_step_free (s);
	
}

void divide_cells(int gen) {
	// Evolve the system as each cell grows by division.
	double y[5000];
	int dof;
	
	string my_name = "output/biofilm";
	ofstream myfile;
	string my_mono_name = my_name+"-line.txt";
	myfile.open(my_mono_name, ios::app);
	
	unsigned long dim = DOF_DIV * cells.size();
	
	const gsl_odeiv2_step_type * TT = gsl_odeiv2_step_rk8pd;

	  gsl_odeiv2_step * s 
	    = gsl_odeiv2_step_alloc (TT, dim);
	  gsl_odeiv2_control * c 
	    = gsl_odeiv2_control_y_new (1e-6, 0.0);
	  gsl_odeiv2_evolve * e 
	    = gsl_odeiv2_evolve_alloc (dim);

	  gsl_odeiv2_system sys = {div_func, nullptr, dim, nullptr};
	  
  	dof = 0;
  	for (int j = 0; j != cells.size(); j++) {
  		y[dof+0] = cells[j]->get_x();
  		y[dof+1] = cells[j]->get_y();
  		y[dof+2] = cells[j]->get_z();
		
		//if (xy == 1) {
		//	y[dof+2] = 0;
		//}
		
  		y[dof+3] = 0; // "initial condition" for torque is always zero
		y[dof+4] = 0;
		y[dof+5] = 0;
		y[dof+6] = 0;
		
		cells[j]->set_l(2);
		
  		y[dof+7] = cells[j]->get_vol();
		
  		dof = dof + DOF_DIV;
  	}
  
  double t = 0.0, tcurrent = tstep;
  double h = 1e-6;
	// Evolve the differential equation
    while (t < T2) {
		
		while (t < tcurrent) {
			
		  	dof = 0;
		  	for (int j = 0; j != cells.size(); j++) {
				
				/*
		  		y[dof+0] = cells[j]->get_x();
		  		y[dof+1] = cells[j]->get_y();
		  		y[dof+2] = cells[j]->get_z();
		
				if (xy == 1) {
					y[dof+2] = 0;
				}*/
				
				//cout << cells[j]->get_d() << endl;
		  		y[dof+3] = 0; // "initial condition" for torque is always zero
		  		y[dof+4] = 0; // "initial condition" for torque is always zero
		  		y[dof+5] = 0; // "initial condition" for torque is always zero
		  		y[dof+6] = 0; // "initial condition" for torque is always zero
				
				//y[dof+7] = cells[j]->get_vol();
		  		dof = dof + DOF_DIV;
		  	}
			
			int status = gsl_odeiv2_evolve_apply (e, c, s,
			                                           &sys, 
			                                           &t, tcurrent,
			                                           &h, y);
													   
			dof = 0;
			for (int j = 0; j != cells.size(); j++) {
				double tx, ty, tz, t_net, t_int, nxi, nyi, nzi, nm;
				
				//cout << "Y DOF: " << y[dof] << " " << y[dof+1] << " " << y[dof+2] << endl;
				
				cells[j]->set_pos(y[dof+0],y[dof+1],y[dof+2]);
				
			  	// Theta axis 1
				nxi = cells[j]->get_nx1();
				nyi = cells[j]->get_ny1();
				nzi = cells[j]->get_nz1();
			  	tx = net_torque(j, 1, 0) * cells[j]->get_theta_axis(0);
			  	ty = net_torque(j, 1, 1) * cells[j]->get_theta_axis(1);
			  	tz = net_torque(j, 1, 2) * cells[j]->get_theta_axis(2);
			  	t_int = cells[j]->get_internal_torque();
			  	t_net = -(sqrt(tx*tx + ty*ty + tz*tz) + t_int);
				if (t_net > 0) {
					cells[j]->set_n1(
									rot(nxi, nyi, nzi, tx/t_net, ty/t_net, tz/t_net, y[dof+3], 0),
									rot(nxi, nyi, nzi, tx/t_net, ty/t_net, tz/t_net, y[dof+3], 1),
									rot(nxi, nyi, nzi, tx/t_net, ty/t_net, tz/t_net, y[dof+3], 2)
								   );
				}
				nxi = cells[j]->get_nx1() + rand()/(100000*static_cast<double>(RAND_MAX));
				nyi = cells[j]->get_ny1() + rand()/(100000*static_cast<double>(RAND_MAX));
				nzi = cells[j]->get_nz1() + rand()/(100000*static_cast<double>(RAND_MAX));
				if (xy == 1) {
					nzi = 0;
				}
				nm = sqrt(nxi*nxi + nyi*nyi + nzi*nzi);
				cells[j]->set_n1(nxi/nm, nyi/nm, nzi/nm); // Normalize the orientation vector, to be sure...
			  
		  	    // Theta axis 2
				nxi = cells[j]->get_nx2();
				nyi = cells[j]->get_ny2();
				nzi = cells[j]->get_nz2();
		  	    tx = net_torque(j, 2, 0) * cells[j]->get_theta_axis(0);
		  	    ty = net_torque(j, 2, 1) * cells[j]->get_theta_axis(1);
		  	    tz = net_torque(j, 2, 2) * cells[j]->get_theta_axis(2);
		  	    t_int = cells[j]->get_internal_torque();
		  	    t_net = sqrt(tx*tx + ty*ty + tz*tz) - t_int;
				if (t_net > 0) {
					cells[j]->set_n2(
									rot(nxi, nyi, nzi, tx/t_net, ty/t_net, tz/t_net, y[dof+4], 0),
									rot(nxi, nyi, nzi, tx/t_net, ty/t_net, tz/t_net, y[dof+4], 1),
									rot(nxi, nyi, nzi, tx/t_net, ty/t_net, tz/t_net, y[dof+4], 2)
								   );
				}
				nxi = cells[j]->get_nx2() + rand()/(100000*static_cast<double>(RAND_MAX));
				nyi = cells[j]->get_ny2() + rand()/(100000*static_cast<double>(RAND_MAX));
				nzi = cells[j]->get_nz2() + rand()/(100000*static_cast<double>(RAND_MAX));
				if (xy == 1) {
					nzi = 0;
				}
				nm = sqrt(nxi*nxi + nyi*nyi + nzi*nzi);
				cells[j]->set_n2(nxi/nm, nyi/nm, nzi/nm); // Normalize the orientation vector, to be sure...
			  
				/*
		  	    // n1 axis
				nxi = cells[j]->get_nx1();
				nyi = cells[j]->get_ny1();
				nzi = cells[j]->get_nz1();
		  	    tx = net_torque(j, 1, 0) * cells[j]->get_n1_axis(0);
		  	    ty = net_torque(j, 1, 1) * cells[j]->get_n1_axis(1);
		  	    tz = net_torque(j, 1, 2) * cells[j]->get_n1_axis(2);
		  	    t_net = sqrt(tx*tx + ty*ty + tz*tz);
				if (t_net > 0) {
					cells[j]->set_n1(
									rot(nxi, nyi, nzi, tx/t_net, ty/t_net, tz/t_net, y[dof+5], 0),
									rot(nxi, nyi, nzi, tx/t_net, ty/t_net, tz/t_net, y[dof+5], 1),
									rot(nxi, nyi, nzi, tx/t_net, ty/t_net, tz/t_net, y[dof+5], 2)
								   );
				}
				nxi = cells[j]->get_nx1() + rand()/(100000*static_cast<double>(RAND_MAX));
				nyi = cells[j]->get_ny1() + rand()/(100000*static_cast<double>(RAND_MAX));
				nzi = cells[j]->get_nz1() + rand()/(100000*static_cast<double>(RAND_MAX));
				if (xy == 1) {
					nzi = 0;
				}
				nm = sqrt(nxi*nxi + nyi*nyi + nzi*nzi);
				cells[j]->set_n1(nxi/nm, nyi/nm, nzi/nm); // Normalize the orientation vector, to be sure...
		  	  
		  	    // n2 axis
				nxi = cells[j]->get_nx2();
				nyi = cells[j]->get_ny2();
				nzi = cells[j]->get_nz2();
		  	    tx = net_torque(j, 2, 0) * cells[j]->get_n2_axis(0);
		  	    ty = net_torque(j, 2, 1) * cells[j]->get_n2_axis(1);
		  	    tz = net_torque(j, 2, 2) * cells[j]->get_n2_axis(2);
		  	    t_net = sqrt(tx*tx + ty*ty + tz*tz);
				if (t_net > 0) {
					cells[j]->set_n2(
									rot(nxi, nyi, nzi, tx/t_net, ty/t_net, tz/t_net, y[dof+6], 0),
									rot(nxi, nyi, nzi, tx/t_net, ty/t_net, tz/t_net, y[dof+6], 1),
									rot(nxi, nyi, nzi, tx/t_net, ty/t_net, tz/t_net, y[dof+6], 2)
								   );
				}
				nxi = cells[j]->get_nx2() + rand()/(100000*static_cast<double>(RAND_MAX));
				nyi = cells[j]->get_ny2() + rand()/(100000*static_cast<double>(RAND_MAX));
				nzi = cells[j]->get_nz2() + rand()/(100000*static_cast<double>(RAND_MAX));
				if (xy == 1) {
					nzi = 0;
				}
				nm = sqrt(nxi*nxi + nyi*nyi + nzi*nzi);
				cells[j]->set_n2(nxi/nm, nyi/nm, nzi/nm); // Normalize the orientation vector, to be sure...
			  	
				*/
				cells[j]->set_d(vol_to_d(y[dof+7]));

				dof = dof + DOF_DIV;
			}
		
			 if (status != GSL_SUCCESS) break;
			
		 }
		 
		myfile << gen*(T1+T2) + T1+t << " " << 2*cells.size() << endl;
		dof = 0;
		for (int j = 0; j != cells.size(); j++) {
			myfile << cells[j]->get_daughter(1)->get_x() << " " << cells[j]->get_daughter(1)->get_y() << " " << cells[j]->get_daughter(1)->get_z() << " "
					<< cells[j]->get_daughter(1)->get_nx() << " " << cells[j]->get_daughter(1)->get_ny() << " " << cells[j]->get_daughter(1)->get_nz() << " "
					<< cells[j]->get_daughter(1)->get_l() << endl;
			myfile << cells[j]->get_daughter(2)->get_x() << " " << cells[j]->get_daughter(2)->get_y() << " " << cells[j]->get_daughter(2)->get_z() << " "
					<< cells[j]->get_daughter(2)->get_nx() << " " << cells[j]->get_daughter(2)->get_ny() << " " << cells[j]->get_daughter(2)->get_nz() << " "
					<< cells[j]->get_daughter(2)->get_l() << endl;
			dof = dof + DOF_GROW;
		}
	
		 tcurrent = tcurrent + tstep;
			 
    }
	
	gsl_odeiv2_evolve_free (e);
	  gsl_odeiv2_control_free (c);
	  gsl_odeiv2_step_free (s);
	
}

void split_cells() {
	// Divide the cells
	int Ni = cells.size();
	
	// add next generation of daughter cells to the list
	for (int j = 0; j != Ni; j++) {
		Cell* c1 = cells[j]->get_daughter(1);
		Cell* c2 = cells[j]->get_daughter(2);
		cells.push_back(new Body(c1->get_x(), c1->get_y(), c1->get_z(), c1->get_nx(), c1->get_ny(), c1->get_nz()));
		cells.push_back(new Body(c2->get_x(), c2->get_y(), c2->get_z(), c2->get_nx(), c2->get_ny(), c2->get_nz()));
	}
	
	// delete the previous generation of mother cells
	cells.erase(cells.begin(), cells.begin()+Ni);
}

int main(int argc, char * argv[]) {
	cout << "\n\n\n\n\n\n\n";
	cout << "Program running\n";
	
	int trial = atoi(argv[1]);  //Optional input for seed
	srand ((trial+1)*time(NULL));
	
	// Initializations:
	cells.clear();
	cells.reserve(3500);
	
	double init_z = 0;
	double init_nx = 1;
	double init_nz = 0;
	
	// Create first cell:
	cells.push_back(new Body(0, 0, init_z, init_nx, 0, init_nz));
	cout << "Hello world." << endl;
	
	// Output data
	string my_name = "output/biofilm";
	ofstream myfile;
	string my_mono_name = my_name+"-line.txt";
	myfile.open(my_mono_name);
	
	myfile << 0 << " " << cells.size() << endl;
	myfile << 0 << " " << 0 << " " << init_z << " "
			<< init_nx << " " << 0 << " " << init_nz << " "
			<< 1 << endl;
	
	for (int gen = 0; gen != maxgen; gen++) {
		//double tcurrent = 0;
		
		cout << "Gen: " << gen << endl;
		
		grow_cells(gen);
		divide_cells(gen);
		split_cells();
		
	}
	
	cout << "Done" << endl;
	
	return 0;
}


