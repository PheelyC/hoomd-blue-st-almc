// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef _SHAPE_MOVES_H
#define _SHAPE_MOVES_H

#include "GSDHPMCSchema.h"
#include "Moves.h"
#include "ShapeUtils.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/extern/quickhull/QuickHull.hpp"
#include <Eigen/Dense>
#include <hoomd/Variant.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "hoomd/extern/clustering/include/clustering/dbscan.h"
#include "hoomd/hpmc/ShapeConvexPolyhedron.h"

namespace hoomd
    {

namespace hpmc
    {

template<typename Shape> class ShapeMoveBase
    {
    public:
    ShapeMoveBase(std::shared_ptr<SystemDefinition> sysdef,
                  std::shared_ptr<IntegratorHPMCMono<Shape>> mc)
        : m_mc(mc), m_sysdef(sysdef)
        {
        m_ntypes = this->m_sysdef->getParticleData()->getNTypes();
        m_volume.resize(m_ntypes, 0);
        m_step_size.resize(m_ntypes, 0);
        }

    virtual ~ShapeMoveBase() {};

    //! prepare is called at the beginning of every update()
    virtual void prepare(uint64_t timestep) { }

    //! construct is called for each particle type that will be changed in update()
    virtual void update_shape(uint64_t,
                              const unsigned int&,
                              typename Shape::param_type&,
                              hoomd::RandomGenerator&)
        {
        }

    //! retreat whenever the proposed move is rejected.
    virtual void retreat(uint64_t timestep, unsigned int type) { }

    Scalar getStepSize(std::string typ)
        {
        unsigned int typid = getValidateType(typ);
        return this->m_step_size[typid];
        }

    void setStepSize(std::string typ, Scalar volume)
        {
        unsigned int typid = getValidateType(typ);
        this->m_step_size[typid] = volume;
        }

    Scalar getVolume(std::string typ)
        {
        unsigned int typid = getValidateType(typ);
        return this->m_volume[typid];
        }

    void setVolume(std::string typ, Scalar volume)
        {
        unsigned int typid = getValidateType(typ);
        this->m_volume[typid] = volume;
        }

    unsigned int getValidateType(std::string typ)
        {
        unsigned int typid = this->m_sysdef->getParticleData()->getTypeByName(typ);
        if (typid >= this->m_ntypes)
            {
            throw std::runtime_error("Invalid particle type.");
            }
        return typid;
        }

    Scalar getMoveProbability()
        {
        return this->m_move_probability;
        }

    void setMoveProbability(Scalar move_probability)
        {
        this->m_move_probability = fmin(move_probability, 1.0);
        }

    virtual Scalar computeLogBoltmann(uint64_t timestep,
                                      const unsigned int& N,
                                      const unsigned int type_id,
                                      const typename Shape::param_type& shape_new,
                                      const Scalar& inew,
                                      const typename Shape::param_type& shape_old,
                                      const Scalar& iold)
        {
        return (Scalar(N) / Scalar(2.0)) * log(inew / iold);
        }

    virtual Scalar computeEnergy(uint64_t timestep,
                                 const unsigned int& N,
                                 const unsigned int type_id,
                                 const typename Shape::param_type& shape,
                                 const Scalar& inertia)
        {
        return 0.0;
        }

    protected:
    std::shared_ptr<IntegratorHPMCMono<Shape>> m_mc;
    std::shared_ptr<SystemDefinition> m_sysdef;
    unsigned m_ntypes;
    std::vector<Scalar> m_volume;
    std::vector<Scalar> m_step_size;
    Scalar m_move_probability;
    std::vector<vec3<Scalar>> m_centroids;

    }; // end class ShapeMoveBase

template<typename Shape> class PythonShapeMove : public ShapeMoveBase<Shape>
    {
    public:
    PythonShapeMove(std::shared_ptr<SystemDefinition> sysdef,
                    std::shared_ptr<IntegratorHPMCMono<Shape>> mc)
        : ShapeMoveBase<Shape>(sysdef, mc)
        {
        m_params.resize(this->m_ntypes);
        m_params_backup.resize(this->m_ntypes);
        }

    void prepare(uint64_t timestep)
        {
        m_params_backup = m_params;
        }

    void update_shape(uint64_t timestep,
                      const unsigned int& type_id,
                      typename Shape::param_type& shape,
                      hoomd::RandomGenerator& rng)
        {
        for (unsigned int i = 0; i < m_params[type_id].size(); i++)
            {
            Scalar stepsize = this->m_step_size[type_id];
            hoomd::UniformDistribution<Scalar> uniform(-stepsize, stepsize);
            Scalar r = hoomd::detail::generate_canonical<double>(rng);
            Scalar x = (r < this->m_move_probability) ? uniform(rng) : 0.0;
            // Reflect trial moves about boundaries
            if (m_params[type_id][i] + x > 1)
                {
                m_params[type_id][i] = 2 - (m_params[type_id][i] + x);
                }
            else if (m_params[type_id][i] + x < 0)
                {
                m_params[type_id][i] = -(m_params[type_id][i] + x);
                }
            else
                {
                m_params[type_id][i] += x;
                }
            }
        pybind11::object d = m_python_callback(type_id, m_params[type_id]);
        pybind11::dict shape_dict = pybind11::cast<pybind11::dict>(d);
        shape = typename Shape::param_type(shape_dict);
        }

    void retreat(uint64_t timestep, unsigned int type)
        {
        // move has been rejected.
        m_params[type] = m_params_backup[type];
        }

    pybind11::list getParams(std::string typ)
        {
        unsigned int type_id = this->getValidateType(typ);
        pybind11::list ret;
        for (unsigned int i = 0; i < m_params[type_id].size(); i++)
            {
            ret.append(m_params[type_id][i]);
            }
        return ret;
        }

    void setParams(std::string typ, pybind11::list params)
        {
        unsigned int type_id = this->getValidateType(typ);
        auto N = pybind11::len(params);
        m_params[type_id].resize(N);
        m_params_backup[type_id].resize(N);
        for (unsigned int i = 0; i < N; i++)
            {
            m_params[type_id][i] = params[i].cast<Scalar>();
            m_params_backup[type_id][i] = params[i].cast<Scalar>();
            }
        }

    pybind11::object getCallback()
        {
        return m_python_callback;
        }

    void setCallback(pybind11::object python_callback)
        {
        m_python_callback = python_callback;
        }

    private:
    std::vector<std::vector<Scalar>>
        m_params_backup;                       // tunable shape parameters to perform trial moves on
    std::vector<std::vector<Scalar>> m_params; // tunable shape parameters to perform trial moves on
    // callback that takes m_params as an argiment and returns a Python dictionary with the shape
    // params.
    pybind11::object m_python_callback;
    };

class ConvexPolyhedronVertexShapeMove : public ShapeMoveBase<ShapeConvexPolyhedron>
    {
    public:
    typedef typename ShapeConvexPolyhedron::param_type param_type;

    ConvexPolyhedronVertexShapeMove(std::shared_ptr<SystemDefinition> sysdef,
                                    std::shared_ptr<IntegratorHPMCMono<ShapeConvexPolyhedron>> mc)
        : ShapeMoveBase<ShapeConvexPolyhedron>(sysdef, mc)
        {
        this->m_centroids.resize(this->m_ntypes, vec3<Scalar>(0, 0, 0));
        initializeMassProperties();
        }

    void initializeMassProperties()
        {
        auto& mc_params = this->m_mc->getParams();
        for (unsigned int i = 0; i < this->m_ntypes; i++)
            {
            detail::MassProperties<ShapeConvexPolyhedron> mp(mc_params[i]);
            this->m_centroids[i] = mp.getCenterOfMass();
            }
        }

    void setVolume(std::string typ, Scalar volume)
        {
        unsigned int typid = getValidateType(typ);
        this->m_volume[typid] = volume;
        auto& mc_params = this->m_mc->getParams();
        param_type shape = mc_params[typid];
        detail::MassProperties<ShapeConvexPolyhedron> mp(shape);
        Scalar current_volume = mp.getVolume();
        vec3<Scalar> dr = this->m_centroids[typid] - mp.getCenterOfMass();
        ShortReal scale = (ShortReal)fast::pow(volume / current_volume, 1.0 / 3.0);
        this->scaleParticleVolume(shape, dr, scale);
        this->m_mc->setParam(typid, shape);
        }

    void scaleParticleVolume(param_type& shape, vec3<Scalar> dr, ShortReal scale)
        {
        Scalar rsq = 0.0;
        for (unsigned int i = 0; i < shape.N; i++)
            {
            shape.x[i] += static_cast<ShortReal>(dr.x);
            shape.x[i] *= scale;
            shape.y[i] += static_cast<ShortReal>(dr.y);
            shape.y[i] *= scale;
            shape.z[i] += static_cast<ShortReal>(dr.z);
            shape.z[i] *= scale;
            vec3<Scalar> vert(shape.x[i], shape.y[i], shape.z[i]);
            rsq = fmax(rsq, dot(vert, vert));
            }
        shape.diameter = ShortReal(2.0 * fast::sqrt(rsq));
        }

    void prepare(uint64_t timestep)
        {
        m_step_size_backup = this->m_step_size;
        }

    void update_shape(uint64_t timestep,
                      const unsigned int& type_id,
                      param_type& shape,
                      hoomd::RandomGenerator& rng)
        {
        // perturb the shape.
        for (unsigned int i = 0; i < shape.N; i++)
            {
            if (hoomd::detail::generate_canonical<double>(rng) < this->m_move_probability)
                {
                vec3<Scalar> vert(shape.x[i], shape.y[i], shape.z[i]);
                move_translate(vert, rng, this->m_step_size[type_id], 3);
                shape.x[i] = static_cast<ShortReal>(vert.x);
                shape.y[i] = static_cast<ShortReal>(vert.y);
                shape.z[i] = static_cast<ShortReal>(vert.z);
                }
            }
        detail::MassProperties<ShapeConvexPolyhedron> mp(shape);
        Scalar volume = mp.getVolume();
        vec3<Scalar> dr = this->m_centroids[type_id] - mp.getCenterOfMass();
        ShortReal scale
            = static_cast<ShortReal>(fast::pow(this->m_volume[type_id] / volume, 1.0 / 3.0));
        scaleParticleVolume(shape, dr, scale);
        this->m_step_size[type_id] *= scale;
        }

    void retreat(uint64_t timestep, unsigned int type)
        {
        this->m_step_size[type] = m_step_size_backup[type];
        }

    private:
    std::vector<Scalar> m_step_size_backup;
    }; // end class ConvexPolyhedronVertexShapeMove

template<class Shape> class ElasticShapeMoveBase : public ShapeMoveBase<Shape>
    {
    public:
    ElasticShapeMoveBase(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<IntegratorHPMCMono<Shape>> mc)
        : ShapeMoveBase<Shape>(sysdef, mc)
        {
        }

    typedef typename Shape::param_type param_type;

    void setStiffness(std::shared_ptr<Variant> stiff)
        {
        m_k = stiff;
        }

    std::shared_ptr<Variant> getStiffness() const
        {
        return m_k;
        }

    pybind11::dict getReferenceShape(std::string typ)
        {
        unsigned int typid = this->getValidateType(typ);
        return m_reference_shapes[typid].asDict();
        }

    virtual void setReferenceShape(std::string typ, pybind11::dict v) = 0;

    protected:
    std::shared_ptr<Variant> m_k; // shape move stiffness
    // shape to reference shape move against
    std::vector<param_type, hoomd::detail::managed_allocator<param_type>> m_reference_shapes;
    };

template<class Shape> class ElasticShapeMove : public ElasticShapeMoveBase<Shape>
    {
    };

template<>
class ElasticShapeMove<ShapeConvexPolyhedron> : public ElasticShapeMoveBase<ShapeConvexPolyhedron>
    {
    public:
    typedef typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixSD;
    typedef typename Eigen::Matrix<Scalar, 3, 3> Matrix3S;

    ElasticShapeMove(std::shared_ptr<SystemDefinition> sysdef,
                     std::shared_ptr<IntegratorHPMCMono<ShapeConvexPolyhedron>> mc)
        : ElasticShapeMoveBase<ShapeConvexPolyhedron>(sysdef, mc)
        {
        m_reference_shapes.resize(this->m_ntypes);
        m_F.resize(this->m_ntypes, Matrix3S::Identity());
        m_F_last.resize(this->m_ntypes, Matrix3S::Identity());
        }

    void prepare(uint64_t timestep)
        {
        m_F_last = m_F;
        }

    //! construct is called at the beginning of every update()
    void update_shape(uint64_t timestep,
                      const unsigned int& type_id,
                      param_type& param,
                      hoomd::RandomGenerator& rng)
        {
        Matrix3S F_curr;
        // perform a scaling move
        if (hoomd::detail::generate_canonical<double>(rng) < this->m_move_probability)
            {
            generateExtentional(F_curr, rng, this->m_step_size[type_id] + 1.0);
            }
        else // perform a rotation-scale-rotation move
            {
            quat<Scalar> q(1.0, vec3<Scalar>(0.0, 0.0, 0.0));
            move_rotate<3>(q, rng, 0.5);
            Matrix3S rot, rot_inv, scale;
            Eigen::Quaternion<double> eq(q.s, q.v.x, q.v.y, q.v.z);
            rot = eq.toRotationMatrix();
            rot_inv = rot.transpose();
            generateExtentional(scale, rng, this->m_step_size[type_id] + 1.0);
            F_curr = rot * scale * rot_inv;
            }
        m_F[type_id] = F_curr * m_F[type_id];
        auto transform = F_curr.cast<ShortReal>();
        Scalar dsq = 0.0;
        for (unsigned int i = 0; i < param.N; i++)
            {
            vec3<ShortReal> vert(param.x[i], param.y[i], param.z[i]);
            param.x[i]
                = transform(0, 0) * vert.x + transform(0, 1) * vert.y + transform(0, 2) * vert.z;
            param.y[i]
                = transform(1, 0) * vert.x + transform(1, 1) * vert.y + transform(1, 2) * vert.z;
            param.z[i]
                = transform(2, 0) * vert.x + transform(2, 1) * vert.y + transform(2, 2) * vert.z;
            vert = vec3<Scalar>(param.x[i], param.y[i], param.z[i]);
            dsq = fmax(dsq, dot(vert, vert));
            }
        param.diameter = ShortReal(2.0 * fast::sqrt(dsq));
        }

    Matrix3S getEps(unsigned int type_id)
        {
        return 0.5 * (m_F[type_id].transpose() + m_F[type_id]) - Matrix3S::Identity();
        }

    Matrix3S getEpsLast(unsigned int type_id)
        {
        return 0.5 * (m_F_last[type_id].transpose() + m_F_last[type_id]) - Matrix3S::Identity();
        }

    //! retreat whenever the proposed move is rejected.
    void retreat(uint64_t timestep, unsigned int type)
        {
        // we can swap because m_F_last will be reset on the next prepare
        m_F[type] = m_F_last[type];
        }

    void setReferenceShape(std::string typ, pybind11::dict v)
        {
        unsigned int typid = this->getValidateType(typ);
        param_type shape = param_type(v, false);
        auto current_shape = this->m_mc->getParams()[typid];
        if (current_shape.N != shape.N)
            {
            throw std::runtime_error(
                "Reference and integrator shapes must have the name number of vertices.");
            }

        m_reference_shapes[typid] = shape;

        // put vertices into matrix form and leverage Eigein's linear algebra
        // tools to solve Vprime = F * Vref for F where:
        //   Vref: vertices of the reference (undeformed) shape (3, N)
        //   Vprime: vertices of the current (deformed) shape (3, N)
        //   F: deformation gradient tensor (3,3)
        MatrixSD Vref(3, shape.N), Vprime(3, shape.N);
        for (unsigned int i = 0; i < shape.N; i++)
            {
            Vref(0, i) = m_reference_shapes[typid].x[i];
            Vref(1, i) = m_reference_shapes[typid].y[i];
            Vref(2, i) = m_reference_shapes[typid].z[i];
            Vprime(0, i) = current_shape.x[i];
            Vprime(1, i) = current_shape.y[i];
            Vprime(2, i) = current_shape.z[i];
            }

        // solve system
        Matrix3S ret = Vref.transpose()
                           .bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
                           .solve(Vprime.transpose());
        m_F[typid] = ret.transpose();

        // compute and store volume of reference shape
        detail::MassProperties<ShapeConvexPolyhedron> mp(m_reference_shapes[typid]);
        this->m_volume[typid] = mp.getVolume();
        }

    Scalar computeLogBoltmann(uint64_t timestep,
                              const unsigned int& N,
                              const unsigned int type_id,
                              const typename ShapeConvexPolyhedron::param_type& shape_new,
                              const Scalar& inew,
                              const typename ShapeConvexPolyhedron::param_type& shape_old,
                              const Scalar& iold)
        {
        Scalar inertia_term = ShapeMoveBase<ShapeConvexPolyhedron>::computeLogBoltmann(timestep,
                                                                                       N,
                                                                                       type_id,
                                                                                       shape_new,
                                                                                       inew,
                                                                                       shape_old,
                                                                                       iold);
        Scalar stiff = (*m_k)(timestep);
        Matrix3S eps = this->getEps(type_id);
        Matrix3S eps_last = this->getEpsLast(type_id);
        Scalar e_ddot_e = (eps * eps).trace();
        Scalar e_ddot_e_last = (eps_last * eps_last).trace();
        return N * stiff * (e_ddot_e_last - e_ddot_e) * this->m_volume[type_id] + inertia_term;
        }

    Scalar computeEnergy(uint64_t timestep,
                         const unsigned int& N,
                         const unsigned int type_id,
                         const param_type& shape,
                         const Scalar& inertia)
        {
        Scalar stiff = (*m_k)(timestep);
        Matrix3S eps = this->getEps(type_id);
        Scalar e_ddot_e = (eps * eps).trace();
        return static_cast<Scalar>(N) * stiff * e_ddot_e * this->m_volume[type_id];
        }

    protected:
    std::vector<Matrix3S> m_F_last; // matrix representing shape deformation at the last step
    std::vector<Matrix3S> m_F;      // matrix representing shape deformation at the current step

    private:
    // These are ElasticShapeMove specific helper functions to randomly
    // sample point on the XYZ=1 surface from a uniform distribution

    //! Check if a point (x,y) lies in the projection of xyz=1 surface
    //! on the xy plane
    inline bool inInSurfaceProjection(Scalar x, Scalar y, Scalar alpha)
        {
        if (x < Scalar(1.0) && y > Scalar(1.0) / (alpha * x))
            return true;
        else if (x >= Scalar(1.0) && y < alpha / x)
            return true;
        else
            return false;
        }

    //! Sample points on the projection of xyz=1
    inline void
    sampleOnSurfaceProjection(Scalar& x, Scalar& y, hoomd::RandomGenerator& rng, Scalar alpha)
        {
        hoomd::UniformDistribution<Scalar> uniform(Scalar(1) / alpha, alpha);
        do
            {
            x = uniform(rng);
            y = uniform(rng);
            } while (!inInSurfaceProjection(x, y, alpha));
        }

    //! Sample points on the projection of xyz=1 surface
    inline void sampleOnSurface(Scalar& x, Scalar& y, hoomd::RandomGenerator& rng, Scalar alpha)
        {
        Scalar sigma_max = 0.0, sigma = 0.0, U = 0.0;
        Scalar alpha2 = alpha * alpha;
        Scalar alpha4 = alpha2 * alpha2;
        sigma_max = fast::sqrt(alpha4 + alpha2 + 1);
        do
            {
            sampleOnSurfaceProjection(x, y, rng, alpha);
            sigma
                = fast::sqrt((1.0 / (x * x * x * x * y * y)) + (1.0 / (x * x * y * y * y * y)) + 1);
            U = hoomd::detail::generate_canonical<Scalar>(rng);
            } while (U > sigma / sigma_max);
        }

    //! Generate an volume conserving extentional deformation matrix
    inline void generateExtentional(Matrix3S& S, hoomd::RandomGenerator& rng, Scalar alpha)
        {
        Scalar x = 0.0, y = 0.0, z = 0.0;
        sampleOnSurface(x, y, rng, alpha);
        z = Scalar(1.0) / x / y;
        S << x, 0.0, 0.0, 0.0, y, 0.0, 0.0, 0.0, z;
        }
    };

template<> class ElasticShapeMove<ShapeEllipsoid> : public ElasticShapeMoveBase<ShapeEllipsoid>
    {
    public:
    typedef typename ShapeEllipsoid::param_type param_type;

    ElasticShapeMove(std::shared_ptr<SystemDefinition> sysdef,
                     std::shared_ptr<IntegratorHPMCMono<ShapeEllipsoid>> mc)
        : ElasticShapeMoveBase<ShapeEllipsoid>(sysdef, mc)
        {
        m_reference_shapes.resize(this->m_ntypes);
        }

    void setReferenceShape(std::string typ, pybind11::dict v)
        {
        unsigned int typid = this->getValidateType(typ);
        m_reference_shapes[typid] = param_type(v, false);
        // compute and store volume of reference shape
        detail::MassProperties<ShapeEllipsoid> mp(m_reference_shapes[typid]);
        this->m_volume[typid] = mp.getVolume();
        }

    void update_shape(uint64_t timestep,
                      const unsigned int& type_id,
                      param_type& param,
                      hoomd::RandomGenerator& rng)
        {
        Scalar lnx = log(param.x / param.y);
        Scalar stepsize = this->m_step_size[type_id];
        Scalar dlnx = hoomd::UniformDistribution<Scalar>(-stepsize, stepsize)(rng);
        Scalar x = fast::exp(lnx + dlnx);
        param.x = static_cast<ShortReal>(x) * param.y;
        param.y = param.y;
        param.z = param.z;
        detail::MassProperties<ShapeEllipsoid> mp(param);
        Scalar volume = mp.getVolume();
        ShortReal scale
            = static_cast<ShortReal>(fast::pow(this->m_volume[type_id] / volume, 1.0 / 3.0));
        param.x *= scale;
        param.y *= scale;
        param.z *= scale;
        }

    void prepare(uint64_t timestep) { }

    void retreat(uint64_t timestep, unsigned int type) { }

    Scalar computeLogBoltmann(uint64_t timestep,
                              const unsigned int& N,
                              const unsigned int type_id,
                              const param_type& shape_new,
                              const Scalar& inew,
                              const param_type& shape_old,
                              const Scalar& iold)
        {
        Scalar inertia_term = ShapeMoveBase<ShapeEllipsoid>::computeLogBoltmann(timestep,
                                                                                N,
                                                                                type_id,
                                                                                shape_new,
                                                                                inew,
                                                                                shape_old,
                                                                                iold);
        Scalar old_energy = computeEnergy(timestep, N, type_id, shape_old, iold);
        Scalar new_energy = computeEnergy(timestep, N, type_id, shape_new, inew);
        return old_energy - new_energy + inertia_term;
        }

    Scalar computeEnergy(uint64_t timestep,
                         const unsigned int& N,
                         const unsigned int type_id,
                         const param_type& shape,
                         const Scalar& inertia)
        {
        Scalar stiff = (*m_k)(timestep);
        Scalar logx = log(shape.x / shape.y);
        return static_cast<Scalar>(N) * stiff * logx * logx;
        }

    private:
    };

template<class Shape> class TargetedShapeMoveBase : public ShapeMoveBase<Shape>
	{
	public:
    TargetedShapeMoveBase(std::shared_ptr<SystemDefinition> sysdef,
                          std::shared_ptr<IntegratorHPMCMono<Shape>> mc)
        : ShapeMoveBase<Shape>(sysdef, mc)
        {
        }

    typedef typename Shape::param_type param_type;

    void setMoveScore(Scalar move_score)
        {
        m_sc = move_score;
        }

    Scalar getMoveScore()
        {
        return m_sc;
        }

	void setTarget1(Scalar t_1)
		{
		m_target_1 = t_1;
		}

	Scalar getTarget1()
		{
		return m_target_1;
		}

	void setTarget2(Scalar t_2)
		{
		m_target_2 = t_2;
		}

	Scalar getTarget2()
		{
		return m_target_2;
		}

	void setScale1(Scalar s_1)
		{
		m_scale_1 = s_1;
		}

	Scalar getScale1()
		{
		return m_scale_1;
		}

	void setScale2(Scalar s_2)
		{
		m_scale_2 = s_2;
		}

	Scalar getScale2()
		{
		return m_scale_2;
		}

	Scalar score(Scalar target_1,
				 Scalar target_2,
				 Scalar param_1,
				 Scalar param_2,
				 Scalar scale_1 = 1,
				 Scalar scale_2 = 1)
		{
		// score shape based on shape parameters
		return (pow(param_1 - target_1, 2) * scale_1) + (pow(param_2 - target_2, 2) * scale_2);
		}

	virtual Scalar computeScore(const typename ShapeConvexPolyhedron::param_type& shape,
								Scalar t_1, Scalar t_2,
								Scalar s_1, Scalar s_2)
		{
		return 0.0;
		}

	void initializeMassProperties()
		{
		auto& mc_params = this->m_mc->getParams();
		for (unsigned int i = 0; i < this->m_ntypes; i++)
			{
			detail::MassProperties<ShapeConvexPolyhedron> mp(mc_params[i]);
			this->m_centroids[i] = mp.getCenterOfMass();
			}
		}

	void setVolume(std::string typ, Scalar volume)
		{
		unsigned int typid = this->getValidateType(typ);
		this->m_volume[typid] = volume;
		auto& mc_params = this->m_mc->getParams();
		param_type shape = mc_params[typid];
		detail::MassProperties<ShapeConvexPolyhedron> mp(shape);
		Scalar current_volume = mp.getVolume();
		vec3<Scalar> dr = this->m_centroids[typid] - mp.getCenterOfMass();
		ShortReal scale = (ShortReal)fast::pow(volume / current_volume, 1.0 / 3.0);
		this->scaleParticleVolume(shape, dr, scale);
		this->m_mc->setParam(typid, shape);
		}

	void scaleParticleVolume(param_type& shape, vec3<Scalar> dr, ShortReal scale)
		{
		Scalar rsq = 0.0;
		for (unsigned int i = 0; i < shape.N; i++)
			{
			shape.x[i] += static_cast<ShortReal>(dr.x);
			shape.x[i] *= scale;
			shape.y[i] += static_cast<ShortReal>(dr.y);
			shape.y[i] *= scale;
			shape.z[i] += static_cast<ShortReal>(dr.z);
			shape.z[i] *= scale;
			vec3<Scalar> vert(shape.x[i], shape.y[i], shape.z[i]);
			rsq = fmax(rsq, dot(vert, vert));
			}
		shape.diameter = ShortReal(2.0 * fast::sqrt(rsq));
		}

	void prepare(uint64_t timestep)
		{
		m_step_size_backup = this->m_step_size;
		}

	void update_shape(uint64_t timestep,
					  const unsigned int& type_id,
					  param_type& shape,
					  hoomd::RandomGenerator& rng)
		{
		// perturb the shape.
		for (unsigned int i = 0; i < shape.N; i++)
			{
			if (hoomd::detail::generate_canonical<double>(rng) < this->m_move_probability)
				{
				vec3<Scalar> vert(shape.x[i], shape.y[i], shape.z[i]);
				move_translate(vert, rng, this->m_step_size[type_id], 3);
				shape.x[i] = static_cast<ShortReal>(vert.x);
				shape.y[i] = static_cast<ShortReal>(vert.y);
				shape.z[i] = static_cast<ShortReal>(vert.z);
				}
			}
		detail::MassProperties<ShapeConvexPolyhedron> mp(shape);
		Scalar volume = mp.getVolume();
		vec3<Scalar> dr = this->m_centroids[type_id] - mp.getCenterOfMass();
		ShortReal scale
			= static_cast<ShortReal>(fast::pow(this->m_volume[type_id] / volume, 1.0 / 3.0));
		scaleParticleVolume(shape, dr, scale);
		this->m_step_size[type_id] *= scale;
		}

	void retreat(uint64_t timestep, unsigned int type)
		{
		this->m_step_size[type] = m_step_size_backup[type];
		}

	Scalar computeLogBoltmann(uint64_t timestep,
							  const unsigned int& N,
							  const unsigned int type_id,
							  const typename ShapeConvexPolyhedron::param_type& shape_new,
							  const Scalar& inew,
							  const typename ShapeConvexPolyhedron::param_type& shape_old,
							  const Scalar& iold)
		{
		Scalar score_pre = computeScore(shape_old,
									    m_target_1, m_target_2,
									    m_scale_1, m_scale_2);
		Scalar score_post = computeScore(shape_new,
										 m_target_1, m_target_2,
										 m_scale_1, m_scale_2);
		setMoveScore(score_pre); // To keep track move score in logger
		return score_pre - score_post;
		}

	protected:
	Scalar m_sc; // shape move score
	Scalar m_target_1;
	Scalar m_target_2;
	Scalar m_scale_1;
	Scalar m_scale_2;
	std::vector<Scalar> m_step_size_backup;
	};

template<class Shape> class Targeted1ShapeMove : public TargetedShapeMoveBase<Shape>
	{
	};

template<>
class Targeted1ShapeMove<ShapeConvexPolyhedron> : public TargetedShapeMoveBase<ShapeConvexPolyhedron>
	{
	public:
	typedef typename ShapeConvexPolyhedron::param_type param_type;

	Targeted1ShapeMove(std::shared_ptr<SystemDefinition> sysdef,
					   std::shared_ptr<IntegratorHPMCMono<ShapeConvexPolyhedron>> mc)
		: TargetedShapeMoveBase<ShapeConvexPolyhedron>(sysdef, mc)
		{
		this->m_centroids.resize(this->m_ntypes, vec3<Scalar>(0, 0, 0));
		initializeMassProperties();
		}

	// Target is Trace of moment of inertia only
	Scalar computeScore(const typename ShapeConvexPolyhedron::param_type& shape,
						Scalar t_1, Scalar t_2,
						Scalar s_1, Scalar s_2)
		{
		detail::MassProperties<ShapeConvexPolyhedron> mp_tar(shape);
		std::vector<Scalar> inert = mp_tar.getInertiaTensor();
		// Compute trace
		Scalar trI = inert[0] + inert[1] + inert[2];
		return score(t_1, 0, trI, 0, s_1, s_2);
		}
	};

template<class Shape> class Targeted2ShapeMove : public TargetedShapeMoveBase<Shape>
	{
	};

template<>
class Targeted2ShapeMove<ShapeConvexPolyhedron> : public TargetedShapeMoveBase<ShapeConvexPolyhedron>
	{
	public:
	typedef typename ShapeConvexPolyhedron::param_type param_type;

	Targeted2ShapeMove(std::shared_ptr<SystemDefinition> sysdef,
					   std::shared_ptr<IntegratorHPMCMono<ShapeConvexPolyhedron>> mc)
		: TargetedShapeMoveBase<ShapeConvexPolyhedron>(sysdef, mc)
		{
		this->m_centroids.resize(this->m_ntypes, vec3<Scalar>(0, 0, 0));
		initializeMassProperties();
		}

	// Target is r_c
	Scalar computeScore(const typename ShapeConvexPolyhedron::param_type& shape,
						Scalar t_1, Scalar t_2,
						Scalar s_1, Scalar s_2)
		{
		detail::MassProperties<ShapeConvexPolyhedron> mp_tar(shape);
		auto p = mp_tar.getQuickHullVertsAndFaces(shape);
        std::vector<vec3<Scalar>> points = p.first;
		Scalar Rc = 0;
		vec3<Scalar> centroid = mp_tar.getCenterOfMass();

		for (unsigned int i = 0; i < points.size(); i++)
		{
			Rc = fmax(Rc, sqrt(pow(points[i][0] - centroid.x, 2) 
				 			 + pow(points[i][1] - centroid.y, 2) 
				 			 + pow(points[i][2] - centroid.z, 2)));
		}
		// std::cout << "Centroid" << "[" << centroid.x << " " << centroid.y << " " << centroid.z << "]\n";
		// std::cout << "R_c:" << Rc << "\n";
		return score(t_1, 0, Rc, 0, s_1, s_2);
		}
	};

template<class Shape> class Targeted1A2ShapeMove : public TargetedShapeMoveBase<Shape>
	{
	};

template<>
class Targeted1A2ShapeMove<ShapeConvexPolyhedron> : public TargetedShapeMoveBase<ShapeConvexPolyhedron>
	{
	public:
	typedef typename ShapeConvexPolyhedron::param_type param_type;

	Targeted1A2ShapeMove(std::shared_ptr<SystemDefinition> sysdef,
					   std::shared_ptr<IntegratorHPMCMono<ShapeConvexPolyhedron>> mc)
		: TargetedShapeMoveBase<ShapeConvexPolyhedron>(sysdef, mc)
		{
		this->m_centroids.resize(this->m_ntypes, vec3<Scalar>(0, 0, 0));
		initializeMassProperties();
		}

	// Target is Tr(I) and R_c, in that order
	Scalar computeScore(const typename ShapeConvexPolyhedron::param_type& shape,
						Scalar t_1, Scalar t_2,
						Scalar s_1, Scalar s_2)
		{
		detail::MassProperties<ShapeConvexPolyhedron> mp_tar(shape);
		std::vector<Scalar> inert = mp_tar.getInertiaTensor();
		// Compute trace
		Scalar trI = inert[0] + inert[1] + inert[2];
		// Compute R_c
		auto p = mp_tar.getQuickHullVertsAndFaces(shape);
        std::vector<vec3<Scalar>> points = p.first;
		Scalar Rc = 0;
		vec3<Scalar> centroid = mp_tar.getCenterOfMass();

		for (unsigned int i = 0; i < points.size(); i++)
		{
			Rc = fmax(Rc, sqrt(pow(points[i][0] - centroid.x, 2) 
				 			 + pow(points[i][1] - centroid.y, 2) 
				 			 + pow(points[i][2] - centroid.z, 2)));
		}
		return score(t_1, t_2, trI, Rc, s_1, s_2);
		}
	};

template<class Shape> class Targeted3ShapeMove : public TargetedShapeMoveBase<Shape>
	{
	};

template<>
class Targeted3ShapeMove<ShapeConvexPolyhedron> : public TargetedShapeMoveBase<ShapeConvexPolyhedron>
	{
	public:
	typedef typename ShapeConvexPolyhedron::param_type param_type;

	Targeted3ShapeMove(std::shared_ptr<SystemDefinition> sysdef,
					   std::shared_ptr<IntegratorHPMCMono<ShapeConvexPolyhedron>> mc)
		: TargetedShapeMoveBase<ShapeConvexPolyhedron>(sysdef, mc)
		{
		this->m_centroids.resize(this->m_ntypes, vec3<Scalar>(0, 0, 0));
		initializeMassProperties();
		}

	// Target is R_i
	Scalar computeScore(const typename ShapeConvexPolyhedron::param_type& shape,
						Scalar t_1, Scalar t_2,
						Scalar s_1, Scalar s_2)
		{
		detail::MassProperties<ShapeConvexPolyhedron> mp_tar(shape);
		vec3<Scalar> centroid = mp_tar.getCenterOfMass();
		// Compute R_i
		Scalar Ri = -666;
		std::vector<std::vector<Scalar>> equations = mp_tar.getEquations();
		for (unsigned int i = 0; i < equations.size(); i++)
		{
			// std::cout << "[" << equations[i][0] << " " << equations[i][1] << " " << equations[i][2] << " " << equations[i][3] << "]\n";
			// Since Ri is negative. Min is negative of max
			Ri = fmax(Ri, mp_tar.getPointPlaneDistance(centroid, equations[i]));	
		}
		// std::cout << "Centroid" << "[" << centroid.x << " " << centroid.y << " " << centroid.z << "]\n";
		// std::cout << "R_i:" << -Ri << "\n";
		return score(t_1, 0, -Ri, 0, s_1, s_2);
		}
	};

template<class Shape> class Targeted4ShapeMove : public TargetedShapeMoveBase<Shape>
	{
	};

template<>
class Targeted4ShapeMove<ShapeConvexPolyhedron> : public TargetedShapeMoveBase<ShapeConvexPolyhedron>
	{
	public:
	typedef typename ShapeConvexPolyhedron::param_type param_type;

	Targeted4ShapeMove(std::shared_ptr<SystemDefinition> sysdef,
					   std::shared_ptr<IntegratorHPMCMono<ShapeConvexPolyhedron>> mc)
		: TargetedShapeMoveBase<ShapeConvexPolyhedron>(sysdef, mc)
		{
		this->m_centroids.resize(this->m_ntypes, vec3<Scalar>(0, 0, 0));
		initializeMassProperties();
		}

	// Target is R_c/R_i
	Scalar computeScore(const typename ShapeConvexPolyhedron::param_type& shape,
						Scalar t_1, Scalar t_2,
						Scalar s_1, Scalar s_2)
		{
		detail::MassProperties<ShapeConvexPolyhedron> mp_tar(shape);
		// Compute R_c
		auto p = mp_tar.getQuickHullVertsAndFaces(shape);
        std::vector<vec3<Scalar>> points = p.first;
		Scalar Rc = 0;
		vec3<Scalar> centroid = mp_tar.getCenterOfMass();
		for (unsigned int i = 0; i < points.size(); i++)
		{
			Rc = fmax(Rc, sqrt(pow(points[i][0] - centroid.x, 2) 
				 			 + pow(points[i][1] - centroid.y, 2) 
				 			 + pow(points[i][2] - centroid.z, 2)));
		}
		// Compute R_i
		Scalar Ri = -666;
		std::vector<std::vector<Scalar>> equations = mp_tar.getEquations();
		for (unsigned int i = 0; i < equations.size(); i++)
		{
			Ri = fmax(Ri, mp_tar.getPointPlaneDistance(centroid, equations[i]));	
		}
		return score(t_1, 0, -Rc/Ri, 0, s_1, s_2);
		}

	private:

	Scalar max_element(std::vector<Scalar> arr)
		{
		Scalar largest = arr[0];
		for (unsigned int i = 1; i < arr.size(); i++)
			{
			if (largest < arr[i]) {
				largest = arr[i];
			}
			}
		return largest;
		}
	};

template<class Shape> class Targeted5ShapeMove : public TargetedShapeMoveBase<Shape>
	{
	};

template<>
class Targeted5ShapeMove<ShapeConvexPolyhedron> : public TargetedShapeMoveBase<ShapeConvexPolyhedron>
	{
	public:
	typedef typename ShapeConvexPolyhedron::param_type param_type;

	Targeted5ShapeMove(std::shared_ptr<SystemDefinition> sysdef,
					   std::shared_ptr<IntegratorHPMCMono<ShapeConvexPolyhedron>> mc)
		: TargetedShapeMoveBase<ShapeConvexPolyhedron>(sysdef, mc)
		{
		this->m_centroids.resize(this->m_ntypes, vec3<Scalar>(0, 0, 0));
		initializeMassProperties();
		}

	// Target is Cos(\theta_d)
	Scalar computeScore(const typename ShapeConvexPolyhedron::param_type& shape,
						Scalar t_1, Scalar t_2,
						Scalar s_1, Scalar s_2)
		{
		return score(t_1, 0, cosine_dihedral(shape), 0, s_1, s_2);
		}

	protected:

	Scalar cosine_dihedral(const typename ShapeConvexPolyhedron::param_type& shape)
	{
		detail::MassProperties<ShapeConvexPolyhedron> mp_tar(shape);
		// Get normal vectors of the faces
		std::vector<std::vector<Scalar>> equations = mp_tar.getEquations();
		std::vector<vec3<Scalar>> normals;
		std::vector<Scalar> sample_weights;
		for (unsigned int i = 0; i < equations.size(); i++)
		{
			normals.push_back({equations[i][0], equations[i][1], equations[i][2]});
		}
		// Running DBSCAN
		NDArray<Scalar, 2> data({normals.size(), 3});
		// Generate data for DBSCAN
		for (unsigned int i = 0; i < normals.size(); i++)
		{
			sample_weights.push_back(10 * mp_tar.getFacetArea(i));
			for (unsigned int j = 0; j < 3; j++)
			{
				data[i][j] = normals[i][j];
			}
		}

		DBSCAN<Scalar> dbscan(data, 0.5, 3, sample_weights);
		dbscan.run();
		const auto &labels = dbscan.labels();
		// std::cout << "Labels size: " << labels.size() << std::endl;
		// std::cout << "Number of clusters: " << dbscan.nClusters() << std::endl;
		// std::cout << "\n[";
		// for (unsigned int j = 0; j < labels.size(); j++)
			// std::cout << labels[j] << " ";
		// std::cout << "]\n";

		// Get faces
		std::vector<std::vector<unsigned int>> faces;
		auto p = mp_tar.getQuickHullVertsAndFaces(shape);
        faces = p.second;
		// Scalar area_cutoff = 0.02;
		int n_clusters = static_cast<unsigned int>(dbscan.nClusters());

		// Compute clustered weighted average normals
		std::vector<vec3<Scalar>> avgs_normal;
		std::vector<Scalar> weights;
		if (n_clusters > 2)
		{
			for (int i = 0; i < n_clusters; i++)
			{
				std::vector<vec3<Scalar>> tmp_norms;
				std::vector<Scalar> tmp_weights;
				Scalar weight_sum = 0.0;
				for (unsigned int j = 0; j < labels.size(); j++)
				{
					if (i == labels[j])
					{
						tmp_norms.push_back(normals[j]);
						Scalar tmp_fA =	mp_tar.getFacetArea(i);
						tmp_weights.push_back(tmp_fA);
						weight_sum += tmp_fA;
						// if (tmp_fA > area_cutoff)
						// {
							// tmp_weights.push_back(tmp_fA);
							// weight_sum += tmp_fA;
						// }
						// else
						// {
							// tmp_weights.push_back(0.0);
						// }
					}
				}
				avgs_normal.push_back(weighted_avg_normals(tmp_norms, tmp_weights));
				weights.push_back(weight_sum);
			}
		}
		else // When not enough clusters are found
		{
			avgs_normal = normals;
			Scalar area_cutoff = 0.02;
			for (unsigned int i = 0; i < avgs_normal.size(); i++)
			{
				Scalar tmp_fA = mp_tar.getFacetArea(i);
				if (tmp_fA > area_cutoff)
				{
					weights.push_back(tmp_fA);
				}
				else
				{
					weights.push_back(0.0);
				}
			}
		}
		// Compute Cosine of dihedral
		Scalar delta = 0.15;
		std::vector<Scalar> cos_d;
		std::vector<Scalar> area_weight;
		for (long unsigned int i = 0; i < avgs_normal.size(); i++)
		{
			for (long unsigned int j = 0; j < avgs_normal.size(); j++)
			{
				Scalar tmp_cosd;
				if (i < j)
				{
					tmp_cosd = dot(avgs_normal[i], avgs_normal[j]) / \
								   sqrt(dot(avgs_normal[i], avgs_normal[i]) * \
								   dot(avgs_normal[j], avgs_normal[j]));
					// Filter out opposite normals
					if (tmp_cosd > -1.0+delta)
					{
						cos_d.push_back(abs(tmp_cosd));
						area_weight.push_back(weights[i] + weights[j]);
					}
					else
					{
						cos_d.push_back(0.01);
						area_weight.push_back(0.0);
					}
				}
			}
		}

		// Any cos_d > delta
		long unsigned int count_accept = std::count_if(cos_d.begin(), cos_d.end(),\
									 				   [delta](Scalar i) {return i > delta;});
		if (count_accept > 0)
		{
			for (unsigned int i = 0; i < cos_d.size(); i++)
			{
				if (cos_d[i] < delta)
					area_weight[i] = 0.0;
			}
		}
		// Check cos_d and weights (very informative for debugging)
		// std::cout << '[';
		// for (unsigned int i = 0; i < cos_d.size(); i++)
		// {
			// std::cout << cos_d[i] << " ";
		// }
		// std::cout << ']';
		// std::cout << std::endl << std::endl;
		// std::cout << '[';
		// for (unsigned int i = 0; i < cos_d.size(); i++)
		// {
			// std::cout << area_weight[i] << " ";
		// }
		// std::cout << ']';
		
		// Get area-weighted cos_d
		std::cout << "\nCos_d avg: " << weighted_avg(cos_d, area_weight) << std::endl;
		return weighted_avg(cos_d, area_weight);
	}

	vec3<Scalar> weighted_avg_normals(std::vector<vec3<Scalar>> norms,
									  std::vector<Scalar> weights)
		{
		vec3<Scalar> norm_tmp = {0.0,0.0,0.0};
		Scalar weight_sum = 0.0;
		for (unsigned int  i = 0; i < norms.size(); i++)
		{
			for (unsigned int j = 0; j < 3; j++)
			{
				norm_tmp[j] += weights[i] * norms[i][j];	
			}
			weight_sum += weights[i];
		}
		for (unsigned int j = 0; j < 3; j++)
		{
			norm_tmp[j] = norm_tmp[j] / weight_sum;	
		}
		return norm_tmp;	
		}

	Scalar weighted_avg(std::vector<Scalar> vals, std::vector<Scalar> weights)
		{
		Scalar avg = 0.0;
		Scalar sum_weight = 0.0;
		for (unsigned int i = 0; i < vals.size(); i++)
		{
			avg += weights[i] * vals[i];
			sum_weight += weights[i];
		}
		return avg / sum_weight; 
		}

	};

template<class Shape> class Targeted1A5ShapeMove : public TargetedShapeMoveBase<Shape>
	{
	};

template<>
class Targeted1A5ShapeMove<ShapeConvexPolyhedron> : public Targeted5ShapeMove<ShapeConvexPolyhedron>
	{
	public:
	typedef typename ShapeConvexPolyhedron::param_type param_type;

	Targeted1A5ShapeMove(std::shared_ptr<SystemDefinition> sysdef,
					   std::shared_ptr<IntegratorHPMCMono<ShapeConvexPolyhedron>> mc)
		: Targeted5ShapeMove<ShapeConvexPolyhedron>(sysdef, mc)
		{
		this->m_centroids.resize(this->m_ntypes, vec3<Scalar>(0, 0, 0));
		initializeMassProperties();
		}

	// Target is Cos(\theta_d) and Tr(I)
	Scalar computeScore(const typename ShapeConvexPolyhedron::param_type& shape,
						Scalar t_1, Scalar t_2,
						Scalar s_1, Scalar s_2)
		{
		detail::MassProperties<ShapeConvexPolyhedron> mp_tar(shape);
		std::vector<Scalar> inert = mp_tar.getInertiaTensor();
		// Compute trace
		return score(t_1, t_2,\
					 inert[0]+inert[1]+inert[2],
					 cosine_dihedral(shape),\
					 s_1, s_2);
		}

	};

namespace detail
    {

template<class Shape> void export_ShapeMoveBase(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<ShapeMoveBase<Shape>, std::shared_ptr<ShapeMoveBase<Shape>>>(m, name.c_str())
        .def("getVolume", &ShapeMoveBase<Shape>::getVolume)
        .def("setVolume", &ShapeMoveBase<Shape>::setVolume)
        .def("getStepSize", &ShapeMoveBase<Shape>::getStepSize)
        .def("setStepSize", &ShapeMoveBase<Shape>::setStepSize);
    }

template<class Shape> void export_PythonShapeMove(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<PythonShapeMove<Shape>,
                     ShapeMoveBase<Shape>,
                     std::shared_ptr<PythonShapeMove<Shape>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<IntegratorHPMCMono<Shape>>>())
        .def_property("param_move_probability",
                      &PythonShapeMove<Shape>::getMoveProbability,
                      &PythonShapeMove<Shape>::setMoveProbability)
        .def("getParams", &PythonShapeMove<Shape>::getParams)
        .def("setParams", &PythonShapeMove<Shape>::setParams)
        .def_property("callback",
                      &PythonShapeMove<Shape>::getCallback,
                      &PythonShapeMove<Shape>::setCallback);
    }

inline void export_ConvexPolyhedronVertexShapeMove(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<ConvexPolyhedronVertexShapeMove,
                     ShapeMoveBase<ShapeConvexPolyhedron>,
                     std::shared_ptr<ConvexPolyhedronVertexShapeMove>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<IntegratorHPMCMono<ShapeConvexPolyhedron>>>())
        .def_property("vertex_move_probability",
                      &ConvexPolyhedronVertexShapeMove::getMoveProbability,
                      &ConvexPolyhedronVertexShapeMove::setMoveProbability)
        .def("getVolume", &ConvexPolyhedronVertexShapeMove::getVolume)
        .def("setVolume", &ConvexPolyhedronVertexShapeMove::setVolume);
    }

template<class Shape>
inline void export_ElasticShapeMove(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<ElasticShapeMove<Shape>,
                     ShapeMoveBase<Shape>,
                     std::shared_ptr<ElasticShapeMove<Shape>>>(m, name.c_str())
        .def_property("normal_shear_ratio",
                      &ElasticShapeMove<Shape>::getMoveProbability,
                      &ElasticShapeMove<Shape>::setMoveProbability)
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<IntegratorHPMCMono<Shape>>>())
        .def_property("stiffness",
                      &ElasticShapeMove<Shape>::getStiffness,
                      &ElasticShapeMove<Shape>::setStiffness)
        .def("setReferenceShape", &ElasticShapeMove<Shape>::setReferenceShape)
        .def("getReferenceShape", &ElasticShapeMove<Shape>::getReferenceShape);
    }

template<class Shape>
inline void export_Targeted1ShapeMove(pybind11::module& m, const std::string& name)
	{
	pybind11::class_<Targeted1ShapeMove<Shape>,
					 ShapeMoveBase<Shape>,
					 std::shared_ptr<Targeted1ShapeMove<Shape>>>(m, name.c_str())
		.def_property("vertex_move_probability",
					  &Targeted1ShapeMove<Shape>::getMoveProbability,
					  &Targeted1ShapeMove<Shape>::setMoveProbability)
		.def(pybind11::init<std::shared_ptr<SystemDefinition>,
							std::shared_ptr<IntegratorHPMCMono<Shape>>>())
		.def_property("move_score",
					  &Targeted1ShapeMove<Shape>::getMoveScore,
					  &Targeted1ShapeMove<Shape>::setMoveScore)
		.def_property("target_1",
					  &Targeted1ShapeMove<Shape>::getTarget1,
					  &Targeted1ShapeMove<Shape>::setTarget1)
		.def_property("target_2",
					  &Targeted1ShapeMove<Shape>::getTarget2,
					  &Targeted1ShapeMove<Shape>::setTarget2)
		.def_property("scale_1",
					  &Targeted1ShapeMove<Shape>::getScale1,
					  &Targeted1ShapeMove<Shape>::setScale1)
		.def_property("scale_2",
					  &Targeted1ShapeMove<Shape>::getScale2,
					  &Targeted1ShapeMove<Shape>::setScale2);
	}

template<class Shape>
inline void export_Targeted2ShapeMove(pybind11::module& m, const std::string& name)
	{
	pybind11::class_<Targeted2ShapeMove<Shape>,
					 ShapeMoveBase<Shape>,
					 std::shared_ptr<Targeted2ShapeMove<Shape>>>(m, name.c_str())
		.def_property("vertex_move_probability",
					  &Targeted2ShapeMove<Shape>::getMoveProbability,
					  &Targeted2ShapeMove<Shape>::setMoveProbability)
		.def(pybind11::init<std::shared_ptr<SystemDefinition>,
							std::shared_ptr<IntegratorHPMCMono<Shape>>>())
		.def_property("move_score",
					  &Targeted2ShapeMove<Shape>::getMoveScore,
					  &Targeted2ShapeMove<Shape>::setMoveScore)
		.def_property("target_1",
					  &Targeted2ShapeMove<Shape>::getTarget1,
					  &Targeted2ShapeMove<Shape>::setTarget1)
		.def_property("target_2",
					  &Targeted2ShapeMove<Shape>::getTarget2,
					  &Targeted2ShapeMove<Shape>::setTarget2)
		.def_property("scale_1",
					  &Targeted2ShapeMove<Shape>::getScale1,
					  &Targeted2ShapeMove<Shape>::setScale1)
		.def_property("scale_2",
					  &Targeted2ShapeMove<Shape>::getScale2,
					  &Targeted2ShapeMove<Shape>::setScale2);
	}

template<class Shape>
inline void export_Targeted1A2ShapeMove(pybind11::module& m, const std::string& name)
	{
	pybind11::class_<Targeted1A2ShapeMove<Shape>,
					 ShapeMoveBase<Shape>,
					 std::shared_ptr<Targeted1A2ShapeMove<Shape>>>(m, name.c_str())
		.def_property("vertex_move_probability",
					  &Targeted1A2ShapeMove<Shape>::getMoveProbability,
					  &Targeted1A2ShapeMove<Shape>::setMoveProbability)
		.def(pybind11::init<std::shared_ptr<SystemDefinition>,
							std::shared_ptr<IntegratorHPMCMono<Shape>>>())
		.def_property("move_score",
					  &Targeted1A2ShapeMove<Shape>::getMoveScore,
					  &Targeted1A2ShapeMove<Shape>::setMoveScore)
		.def_property("target_1",
					  &Targeted1A2ShapeMove<Shape>::getTarget1,
					  &Targeted1A2ShapeMove<Shape>::setTarget1)
		.def_property("target_2",
					  &Targeted1A2ShapeMove<Shape>::getTarget2,
					  &Targeted1A2ShapeMove<Shape>::setTarget2)
		.def_property("scale_1",
					  &Targeted1A2ShapeMove<Shape>::getScale1,
					  &Targeted1A2ShapeMove<Shape>::setScale1)
		.def_property("scale_2",
					  &Targeted1A2ShapeMove<Shape>::getScale2,
					  &Targeted1A2ShapeMove<Shape>::setScale2);
	}

template<class Shape>
inline void export_Targeted3ShapeMove(pybind11::module& m, const std::string& name)
	{
	pybind11::class_<Targeted3ShapeMove<Shape>,
					 ShapeMoveBase<Shape>,
					 std::shared_ptr<Targeted3ShapeMove<Shape>>>(m, name.c_str())
		.def_property("vertex_move_probability",
					  &Targeted3ShapeMove<Shape>::getMoveProbability,
					  &Targeted3ShapeMove<Shape>::setMoveProbability)
		.def(pybind11::init<std::shared_ptr<SystemDefinition>,
							std::shared_ptr<IntegratorHPMCMono<Shape>>>())
		.def_property("move_score",
					  &Targeted3ShapeMove<Shape>::getMoveScore,
					  &Targeted3ShapeMove<Shape>::setMoveScore)
		.def_property("target_1",
					  &Targeted3ShapeMove<Shape>::getTarget1,
					  &Targeted3ShapeMove<Shape>::setTarget1)
		.def_property("target_2",
					  &Targeted3ShapeMove<Shape>::getTarget2,
					  &Targeted3ShapeMove<Shape>::setTarget2)
		.def_property("scale_1",
					  &Targeted3ShapeMove<Shape>::getScale1,
					  &Targeted3ShapeMove<Shape>::setScale1)
		.def_property("scale_2",
					  &Targeted3ShapeMove<Shape>::getScale2,
					  &Targeted3ShapeMove<Shape>::setScale2);
	}

template<class Shape>
inline void export_Targeted4ShapeMove(pybind11::module& m, const std::string& name)
	{
	pybind11::class_<Targeted4ShapeMove<Shape>,
					 ShapeMoveBase<Shape>,
					 std::shared_ptr<Targeted4ShapeMove<Shape>>>(m, name.c_str())
		.def_property("vertex_move_probability",
					  &Targeted4ShapeMove<Shape>::getMoveProbability,
					  &Targeted4ShapeMove<Shape>::setMoveProbability)
		.def(pybind11::init<std::shared_ptr<SystemDefinition>,
							std::shared_ptr<IntegratorHPMCMono<Shape>>>())
		.def_property("move_score",
					  &Targeted4ShapeMove<Shape>::getMoveScore,
					  &Targeted4ShapeMove<Shape>::setMoveScore)
		.def_property("target_1",
					  &Targeted4ShapeMove<Shape>::getTarget1,
					  &Targeted4ShapeMove<Shape>::setTarget1)
		.def_property("target_2",
					  &Targeted4ShapeMove<Shape>::getTarget2,
					  &Targeted4ShapeMove<Shape>::setTarget2)
		.def_property("scale_1",
					  &Targeted4ShapeMove<Shape>::getScale1,
					  &Targeted4ShapeMove<Shape>::setScale1)
		.def_property("scale_2",
					  &Targeted4ShapeMove<Shape>::getScale2,
					  &Targeted4ShapeMove<Shape>::setScale2);
	}

template<class Shape>
inline void export_Targeted5ShapeMove(pybind11::module& m, const std::string& name)
	{
	pybind11::class_<Targeted5ShapeMove<Shape>,
					 ShapeMoveBase<Shape>,
					 std::shared_ptr<Targeted5ShapeMove<Shape>>>(m, name.c_str())
		.def_property("vertex_move_probability",
					  &Targeted5ShapeMove<Shape>::getMoveProbability,
					  &Targeted5ShapeMove<Shape>::setMoveProbability)
		.def(pybind11::init<std::shared_ptr<SystemDefinition>,
							std::shared_ptr<IntegratorHPMCMono<Shape>>>())
		.def_property("move_score",
					  &Targeted5ShapeMove<Shape>::getMoveScore,
					  &Targeted5ShapeMove<Shape>::setMoveScore)
		.def_property("target_1",
					  &Targeted5ShapeMove<Shape>::getTarget1,
					  &Targeted5ShapeMove<Shape>::setTarget1)
		.def_property("target_2",
					  &Targeted5ShapeMove<Shape>::getTarget2,
					  &Targeted5ShapeMove<Shape>::setTarget2)
		.def_property("scale_1",
					  &Targeted5ShapeMove<Shape>::getScale1,
					  &Targeted5ShapeMove<Shape>::setScale1)
		.def_property("scale_2",
					  &Targeted5ShapeMove<Shape>::getScale2,
					  &Targeted5ShapeMove<Shape>::setScale2);
	}

template<class Shape>
inline void export_Targeted1A5ShapeMove(pybind11::module& m, const std::string& name)
	{
	pybind11::class_<Targeted1A5ShapeMove<Shape>,
					 ShapeMoveBase<Shape>,
					 std::shared_ptr<Targeted1A5ShapeMove<Shape>>>(m, name.c_str())
		.def_property("vertex_move_probability",
					  &Targeted1A5ShapeMove<Shape>::getMoveProbability,
					  &Targeted1A5ShapeMove<Shape>::setMoveProbability)
		.def(pybind11::init<std::shared_ptr<SystemDefinition>,
							std::shared_ptr<IntegratorHPMCMono<Shape>>>())
		.def_property("move_score",
					  &Targeted1A5ShapeMove<Shape>::getMoveScore,
					  &Targeted1A5ShapeMove<Shape>::setMoveScore)
		.def_property("target_1",
					  &Targeted1A5ShapeMove<Shape>::getTarget1,
					  &Targeted1A5ShapeMove<Shape>::setTarget1)
		.def_property("target_2",
					  &Targeted1A5ShapeMove<Shape>::getTarget2,
					  &Targeted1A5ShapeMove<Shape>::setTarget2)
		.def_property("scale_1",
					  &Targeted1A5ShapeMove<Shape>::getScale1,
					  &Targeted1A5ShapeMove<Shape>::setScale1)
		.def_property("scale_2",
					  &Targeted1A5ShapeMove<Shape>::getScale2,
					  &Targeted1A5ShapeMove<Shape>::setScale2);
	}

    } // namespace detail
    } // namespace hpmc
    } // namespace hoomd

#endif
