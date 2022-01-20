/*  Multivariate Normal distribution implementation based on Eigen and C++11 random utilities.
    Inspired in:
        https://github.com/beniz/eigenmvn
        https://stackoverflow.com/questions/6142576/sample-from-multivariate-normal-gaussian-distribution-in-c
        http://stackoverflow.com/questions/16361226/error-while-creating-object-from-templated-class
        http://lost-found-wandering.blogspot.fr/2011/05/sampling-from-multivariate-normal-in-c.html
*/

#ifndef FAKE_VEHICLE_DETECTION__MULTIVARIATENORMAL_HPP_
#define FAKE_VEHICLE_DETECTION__MULTIVARIATENORMAL_HPP_

#include <cmath>
#include <random>

#include <Eigen/Dense>

namespace fake_vehicle_detection
{

template<typename Scalar, int Size>
class MultivariateNormal
{
    Eigen::Matrix<Scalar, Size, Size> _covar;
    Eigen::Matrix<Scalar, Size, Size> _transform;
    Eigen::Matrix<Scalar, Size, 1> _mean;

public:
    MultivariateNormal(const Eigen::Matrix<Scalar, Size, 1>& mean,
                       const Eigen::Matrix<Scalar, Size, Size>& covar)
    {
        setMean(mean);
        setCovar(covar);
    }

    void setMean(const Eigen::Matrix<Scalar, Size, 1>& mean)
    {
        _mean = mean;
    }

    void setCovar(const Eigen::Matrix<Scalar, Size, Size>& covar)
    {
        _covar = covar;

        // Assuming that we'll be using this repeatedly,
        // compute the transformation matrix that will
        // be applied to unit-variance independent normals

        // We can only use the cholesky decomposition if 
        // the covariance matrix is symmetric, pos-definite.
        // But a covariance matrix might be pos-semi-definite.
        // In that case, we'll go to an EigenSolver

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, Size, Size> > solver =
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, Size, Size> >(_covar);
        _transform = solver.eigenvectors() * solver.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();
    }

    // Draw one sample
    Eigen::Matrix<Scalar, Size, 1> sample(std::mt19937& rng)
    {
        std::normal_distribution<Scalar> norm;
        Eigen::Matrix<Scalar, Size, 1> R { norm(rng), norm(rng) };
        return (_transform * R) + _mean;
    }
};

} // Namespace fake_vehicle_detection

#endif // FAKE_VEHICLE_DETECTION__MULTIVARIATENORMAL_HPP_
