#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "bool.hpp"

using namespace longshot;
using namespace pybind11::literals;

namespace py = pybind11;

class PyBaseBooleanFunction : public BaseBooleanFunction
{
public:
    using BaseBooleanFunction::BaseBooleanFunction;

    bool eval(BaseBooleanFunction::input_t x) const override {
        PYBIND11_OVERRIDE_PURE(
            bool,
            BaseBooleanFunction,
            eval,
            x
        );
    }
    void as_cnf() override {
        PYBIND11_OVERRIDE_PURE(
            void,
            BaseBooleanFunction,
            as_cnf
        );
    }
    void as_dnf() override {
        PYBIND11_OVERRIDE_PURE(
            void,
            BaseBooleanFunction,
            as_dnf
        );
    }
};


PYBIND11_MODULE(_core, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    py::class_<Literals>(m, "_Literals")
        .def(py::init<uint32_t, uint32_t>())
        .def(py::init<const Literals &>())
        .def(py::init<>())
        .def_property_readonly("is_empty", &Literals::is_empty)
        .def_property_readonly("is_contradictory", &Literals::is_contradictory)
        .def_property_readonly("is_constant", &Literals::is_constant)
        .def_property_readonly("width", &Literals::width)
        .def_property_readonly("pos", &Literals::pos)
        .def_property_readonly("neg", &Literals::neg)
        ;

    py::class_<DecisionTree>(m, "_CppDecisionTree")
        .def(py::init<>())
        .def(py::init<int>())
        .def(py::init<int, const DecisionTree &, const DecisionTree &>())
        .def(py::init<const DecisionTree &>())
        .def("delete", &DecisionTree::delete_tree) // remember to delete the tree at the end
        .def("decide", &DecisionTree::decide) 
        .def_property_readonly("lt", &DecisionTree::ltree)
        .def_property_readonly("rt", &DecisionTree::rtree)
        .def_property_readonly("is_constant", &DecisionTree::is_constant)
        .def_property_readonly("var", &DecisionTree::var)
        ;

    py::class_<BaseBooleanFunction, PyBaseBooleanFunction /* <--- trampoline */>(m, "_BaseBooleanFunction")
        .def(py::init<int>())
        .def("eval", &BaseBooleanFunction::eval)
        .def("as_cnf", &BaseBooleanFunction::as_cnf)
        .def("as_dnf", &BaseBooleanFunction::as_dnf)
        .def("avgQ", &BaseBooleanFunction::avgQ, "tree"_a = nullptr)
        .def_property_readonly("num_vars", &BaseBooleanFunction::num_vars)
        ;

    py::class_<MonotonicBooleanFunction, BaseBooleanFunction>(m, "_MonotonicBooleanFunction")
        .def(py::init<int>())
        .def(py::init<const MonotonicBooleanFunction &>())
        .def("eval", &MonotonicBooleanFunction::eval)
        .def("as_cnf", &MonotonicBooleanFunction::as_cnf)
        .def("as_dnf", &MonotonicBooleanFunction::as_dnf)
        .def("avgQ", &MonotonicBooleanFunction::avgQ, "tree"_a = nullptr)
        .def("add_clause", &MonotonicBooleanFunction::add_clause)
        .def("add_term", &MonotonicBooleanFunction::add_term)
        .def_property_readonly("num_vars", &MonotonicBooleanFunction::num_vars)
        ;

    py::class_<CountingBooleanFunction, BaseBooleanFunction>(m, "_CountingBooleanFunction")
        .def(py::init<int>())
        .def(py::init<const CountingBooleanFunction &>())
        .def("eval", &CountingBooleanFunction::eval)
        .def("as_cnf", &CountingBooleanFunction::as_cnf)
        .def("as_dnf", &CountingBooleanFunction::as_dnf)
        .def("avgQ", &CountingBooleanFunction::avgQ, "tree"_a = nullptr)
        .def("add_clause", &CountingBooleanFunction::add_clause)
        .def("add_term", &CountingBooleanFunction::add_term)
        .def("del_clause", &CountingBooleanFunction::del_clause)
        .def("del_term", &CountingBooleanFunction::del_term)
        .def_property_readonly("num_vars", &CountingBooleanFunction::num_vars)
        ;
}