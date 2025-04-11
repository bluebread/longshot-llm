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

PYBIND11_MODULE(longshot, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
        
    py::class_<AC0_Circuit, PyAC0_Circuit /* <--- trampoline */>(m, "AC0_Circuit")
        .def(py::init<int, int>())
        .def("eval", &AC0_Circuit::eval)
        .def("avgQ", &AC0_Circuit::avgQ)
        .def("is_constant", &AC0_Circuit::is_constant)
        .def_property_readonly("num_vars", &AC0_Circuit::num_vars)
        .def_property_readonly("size", &AC0_Circuit::size)
        .def_property_readonly("depth", &AC0_Circuit::depth)
        ;

    py::class_<NormalFormFormula::Clause>(m, "Clause")
        .def(py::init<longshot::AC0_Circuit::input_t, longshot::AC0_Circuit::input_t>())
        .def(py::init<const py::dict &>())
        .def_readwrite("variables", &NormalFormFormula::Clause::variables)
        .def_readwrite("negated_variables", &NormalFormFormula::Clause::negated_variables)
        ;
    
    py::enum_<NormalFormFormula::Type>(m, "NormalFormFormulaType")
        .value("Conjunctive", NormalFormFormula::Type::Conjunctive)
        .value("Disjunctive", NormalFormFormula::Type::Disjunctive)
        .export_values()
        ;

    py::class_<NormalFormFormula, AC0_Circuit>(m, "NormalFormFormula")
        .def(py::init<int, NormalFormFormula::Type>(), 
             "x"_a, py::pos_only(), "type"_a = NormalFormFormula::Type::Disjunctive)
        .def("add_clause", &NormalFormFormula::add_clause)
        .def("eval", &NormalFormFormula::eval)
        .def("avgQ", &NormalFormFormula::avgQ)
        .def_property_readonly("width", &NormalFormFormula::width)
        .def_property_readonly("clauses", &NormalFormFormula::clauses)
        ;
}