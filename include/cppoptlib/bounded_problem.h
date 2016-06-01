#ifndef BOUNDED_PROBLEM_H
#define BOUNDED_PROBLEM_H

#include "meta.h"

namespace cppoptlib {

    /**
     * @brief   This subclass represents a bounded problem, i.e. a problem containing a lower as well as an upper bound.
     *
     * @note    Partially bounded problems can be specified by setting the respective bound to e.g. 'Vector<T>::Ones(DIM)*std::numeric_limits<T>::infinity()'
     *
     * @code    For example:
     *
     *          template <class T>
     *          class MyPartiallyBoundedProblem : public BoundedProblem<T> {
     *              using super = BoundedProblem<T>;
     *
     *              MyBoundedProblem() { super( -Vector<T>::Ones(DIM)*std::numeric_limits<T>::infinity(), Vector<T>::Ones(DIM) ); }
     *
     *              T value() { ... }
     *          }
     */
    template<typename T>
    class BoundedProblem : public Problem<T> {
        protected:
            Vector<T> lowerBound_;
            Vector<T> upperBound_;

        public:
            BoundedProblem(Vector<T> const& lowerBound, Vector<T> const& upperBound) : lowerBound_(lowerBound), upperBound_(upperBound) {}
            virtual ~BoundedProblem() = default;


            void setLowerBound(Vector<T>  lb) { lowerBound_ = lb; }

            void setUpperBound(Vector<T>  ub) { upperBound_ = ub; }

            void setBoxConstraint(Vector<T>  lb, Vector<T>  ub) {
                setLowerBound(lb);
                setUpperBound(ub);
            }

            Vector<T> lowerBound() const { return lowerBound_; }

            Vector<T> upperBound() const { return upperBound_; }
    };

} // end namespace cppoptlib
















#endif