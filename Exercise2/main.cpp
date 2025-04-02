#include <iostream>
#include "Eigen/Eigen"
using namespace std;
using namespace Eigen;
//Come calcolare l'errore relativo
double errore( const VectorXd& calcolato, const VectorXd& x_esatto){
	return (calcolato - x_esatto).norm()/x_esatto.norm();
}
int main()
{
	//Prima di inserire il valore, bisogna dichiararle
	MatrixXd A1(2,2), A2(2,2), A3(2,2);
	VectorXd x_esatto(2);
	VectorXd b1(2), b2(2), b3(2);
	
	A1 << 5.547001962252291e-01, -3.770900990025203e-02, 8.320502943378437e-01,
	-9.992887623566787e-01;
    A2 << 5.547001962252291e-01, -5.540607316466765e-01, 8.320502943378437e-01,
	-8.324762492991313e-01;
	A3 << 5.547001962252291e-01, -5.547001955851905e-01, 8.320502943378437e-01,
	-8.320502947645361e-01;
	x_esatto << -1.0e+00, -1.0e+00;
	b1<< -5.169911863249772e-01, 1.672384680188350e-01;
	b2<< -6.394645785530173e-04, 4.259549612877223e-04;
	b3<< -6.400391328043042e-10, 4.266924591433963e-10;
	

//PALU
	FullPivLU<MatrixXd> lu1(A1);
	VectorXd x_f1 = lu1.solve(b1);
	FullPivLU<MatrixXd> lu2(A2);
	VectorXd x_f2 = lu2.solve(b2);
	FullPivLU<MatrixXd> lu3(A3);
	VectorXd x_f3 = lu3.solve(b3);
	
	double err1p= errore(x_f1, x_esatto);
	double err2p= errore(x_f2, x_esatto);
	double err3p= errore(x_f3, x_esatto);
	
	cout << "il risultato dell'errore rel di 1 con PALU è:" << err1p << endl;
	cout << "il risultato dell'errore rel di 2 con PALU è:" << err2p << endl;
	cout << "il risultato dell'errore rel di 3 con PALU è:" << err3p << endl;
//QR
    HouseholderQR<MatrixXd> qr1(A1);
	MatrixXd Q1 = qr1.householderQ();
	MatrixXd R1 = qr1.matrixQR().triangularView<Upper>();
	MatrixXd A1_r = Q1*R1;
	
	HouseholderQR<MatrixXd> qr2(A2);
	MatrixXd Q2 = qr2.householderQ();
	MatrixXd R2 = qr2.matrixQR().triangularView<Upper>();
	MatrixXd A2_r = Q2*R2;
	
	HouseholderQR<MatrixXd> qr3(A3);
	MatrixXd Q3 = qr3.householderQ();
	MatrixXd R3 = qr3.matrixQR().triangularView<Upper>();
	MatrixXd A3_r = Q3*R3;
	
	double err1q = (A1 - A1_r).norm()/A1.norm();
	double err2q = (A2 - A2_r).norm()/A2.norm();
	double err3q = (A3 - A3_r).norm()/A3.norm();
	
	cout << "l'errore 1 con QR è:" << err1q << endl;
	cout << "l'errore 2 con QR è:" << err2q << endl;
	cout << "l'errore 3 con QR è:" << err3q << endl;
	
	return 0;
}
