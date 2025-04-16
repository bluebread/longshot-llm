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

    py::class_<NormalFormFormula::Literals>(cm, "_Literals")
        .def(py::init<uint32_t, uint32_t>())
        .def(py::init<const NormalFormFormula::Literals &>())
        .def(py::init<>())
        .def("is_empty", &NormalFormFormula::Literals::is_empty)
        .def("is_contradictory", &NormalFormFormula::Literals::is_contradictory)
        .def("is_constant", &NormalFormFormula::Literals::is_constant)
        .def("width", &NormalFormFormula::Literals::width)
        .def_property_readonly("pos", &NormalFormFormula::Literals::pos)
        .def_property_readonly("neg", &NormalFormFormula::Literals::neg)
        ;
    
    py::enum_<NormalFormFormula::Type>(cm, "_NormalFormFormulaType")
        .value("Conjunctive", NormalFormFormula::Type::Conjunctive)
        .value("Disjunctive", NormalFormFormula::Type::Disjunctive)
        .export_values()
        ;

    py::class_<NormalFormFormula, AC0_Circuit>(cm, "_NormalFormFormula")
        .def(py::init<int, NormalFormFormula::Type>(), 
             "n"_a, py::pos_only(), "type"_a = NormalFormFormula::Type::Disjunctive)
        .def(py::init<const NormalFormFormula &>())
        .def("add", &NormalFormFormula::add)
        .def("eval", &NormalFormFormula::eval)
        .def("avgQ", &NormalFormFormula::avgQ)
        .def_property_readonly("ftype", &NormalFormFormula::ftype)
        .def_property_readonly("width", &NormalFormFormula::width)
        .def_property_readonly("literals", &NormalFormFormula::literals)
        ;
}