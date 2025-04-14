#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "circuit.hpp"

using namespace longshot;
using namespace pybind11::literals;

namespace py = pybind11;

class PyAC0_Circuit : public AC0_Circuit
{
public:
    using AC0_Circuit::AC0_Circuit;

    bool eval (AC0_Circuit::input_t x) const override {
        PYBIND11_OVERRIDE_PURE(
            bool,
            AC0_Circuit,
            eval,
            x
        );
    }

    double avgQ() const override {
        PYBIND11_OVERRIDE_PURE(
            double,
            AC0_Circuit,
            avgQ,
        );
    }

    bool is_constant() const  override {
        PYBIND11_OVERRIDE(
            bool,
            AC0_Circuit,
            is_constant,
        );
    }
};

PYBIND11_MODULE(_core, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    py::module_ cm = m.def_submodule("circuit");
        
    py::class_<AC0_Circuit, PyAC0_Circuit /* <--- trampoline */>(cm, "_AC0_Circuit")
        .def(py::init<int, int>())
        .def("eval", &AC0_Circuit::eval)
        .def("avgQ", &AC0_Circuit::avgQ)
        .def("is_constant", &AC0_Circuit::is_constant)
        .def_property_readonly("num_vars", &AC0_Circuit::num_vars)
        .def_property_readonly("size", &AC0_Circuit::size)
        .def_property_readonly("depth", &AC0_Circuit::depth)
        ;

    py::class_<NormalFormFormula::Clause>(cm, "_Clause")
        .def(py::init<longshot::AC0_Circuit::input_t, longshot::AC0_Circuit::input_t>())
        .def(py::init<const NormalFormFormula::Clause &>())
        .def(py::init<>())
        .def_readwrite("pos_vars", &NormalFormFormula::Clause::pos_vars)
        .def_readwrite("neg_vars", &NormalFormFormula::Clause::neg_vars)
        ;
    
    py::enum_<NormalFormFormula::Type>(cm, "_NormalFormFormulaType")
        .value("Conjunctive", NormalFormFormula::Type::Conjunctive)
        .value("Disjunctive", NormalFormFormula::Type::Disjunctive)
        .export_values()
        ;

    py::class_<NormalFormFormula, AC0_Circuit>(cm, "_NormalFormFormula")
        .def(py::init<int, NormalFormFormula::Type>(), 
             "x"_a, py::pos_only(), "type"_a = NormalFormFormula::Type::Disjunctive)
        .def(py::init<const NormalFormFormula &>())
        .def(py::init<NormalFormFormula &&>())
        .def("add_clause", &NormalFormFormula::add_clause)
        .def("eval", &NormalFormFormula::eval)
        .def("avgQ", &NormalFormFormula::avgQ)
        .def_property_readonly("width", &NormalFormFormula::width)
        .def_property_readonly("clauses", &NormalFormFormula::clauses)
        ;
}