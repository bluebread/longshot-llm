#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

// #include "circuit.hpp"
#include "bool.hpp"

using namespace longshot;
using namespace pybind11::literals;

namespace py = pybind11;

// class PyAC0_Circuit : public AC0_Circuit
// {
// public:
//     using AC0_Circuit::AC0_Circuit;

//     bool eval (AC0_Circuit::input_t x) const override {
//         PYBIND11_OVERRIDE_PURE(
//             bool,
//             AC0_Circuit,
//             eval,
//             x
//         );
//     }

//     double avgQ() const override {
//         PYBIND11_OVERRIDE_PURE(
//             double,
//             AC0_Circuit,
//             avgQ,
//         );
//     }

//     bool is_constant() const  override {
//         PYBIND11_OVERRIDE_PURE(
//             bool,
//             AC0_Circuit,
//             is_constant,
//         );
//     }
// };

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
        .def("is_empty", &Literals::is_empty)
        .def("is_contradictory", &Literals::is_contradictory)
        .def("is_constant", &Literals::is_constant)
        .def("width", &Literals::width)
        .def_property_readonly("pos", &Literals::pos)
        .def_property_readonly("neg", &Literals::neg)
        ;

    py::class_<BaseBooleanFunction, PyBaseBooleanFunction /* <--- trampoline */>(m, "_BaseBooleanFunction")
        .def(py::init<int>())
        .def("eval", &BaseBooleanFunction::eval)
        .def("as_cnf", &BaseBooleanFunction::as_cnf)
        .def("as_dnf", &BaseBooleanFunction::as_dnf)
        .def("avgQ", &BaseBooleanFunction::avgQ)
        .def_property_readonly("num_vars", &BaseBooleanFunction::num_vars)
        ;

    py::class_<MonotonicBooleanFunction, BaseBooleanFunction>(m, "_MonotonicBooleanFunction")
        .def(py::init<int>())
        .def(py::init<const MonotonicBooleanFunction &>())
        .def("eval", &MonotonicBooleanFunction::eval)
        .def("as_cnf", &MonotonicBooleanFunction::as_cnf)
        .def("as_dnf", &MonotonicBooleanFunction::as_dnf)
        .def("avgQ", &MonotonicBooleanFunction::avgQ)
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
        .def("avgQ", &CountingBooleanFunction::avgQ)
        .def("add_clause", &CountingBooleanFunction::add_clause)
        .def("add_term", &CountingBooleanFunction::add_term)
        .def("del_clause", &CountingBooleanFunction::del_clause)
        .def("del_term", &CountingBooleanFunction::del_term)
        .def_property_readonly("num_vars", &CountingBooleanFunction::num_vars)
        ;

    // py::module_ cm = m.def_submodule("circuit");
        
    // py::class_<AC0_Circuit, PyAC0_Circuit /* <--- trampoline */>(cm, "_AC0_Circuit")
    //     .def(py::init<int, int>())
    //     .def("eval", &AC0_Circuit::eval)
    //     .def("avgQ", &AC0_Circuit::avgQ)
    //     .def("is_constant", &AC0_Circuit::is_constant)
    //     .def_property_readonly("num_vars", &AC0_Circuit::num_vars)
    //     .def_property_readonly("size", &AC0_Circuit::size)
    //     .def_property_readonly("depth", &AC0_Circuit::depth)
    //     ;

    // py::class_<NormalFormFormula::Literals>(cm, "_Literals")
    //     .def(py::init<uint32_t, uint32_t>())
    //     .def(py::init<const NormalFormFormula::Literals &>())
    //     .def(py::init<>())
    //     .def("is_empty", &NormalFormFormula::Literals::is_empty)
    //     .def("is_contradictory", &NormalFormFormula::Literals::is_contradictory)
    //     .def("is_constant", &NormalFormFormula::Literals::is_constant)
    //     .def("width", &NormalFormFormula::Literals::width)
    //     .def_property_readonly("pos", &NormalFormFormula::Literals::pos)
    //     .def_property_readonly("neg", &NormalFormFormula::Literals::neg)
    //     ;
    
    // py::enum_<NormalFormFormula::Type>(cm, "_NormalFormFormulaType")
    //     .value("Conjunctive", NormalFormFormula::Type::Conjunctive)
    //     .value("Disjunctive", NormalFormFormula::Type::Disjunctive)
    //     .export_values()
    //     ;

    // py::class_<NormalFormFormula, AC0_Circuit>(cm, "_NormalFormFormula")
    //     .def(py::init<int, NormalFormFormula::Type>(), 
    //          "n"_a, py::pos_only(), "type"_a = NormalFormFormula::Type::Disjunctive)
    //     .def(py::init<const NormalFormFormula &>())
    //     .def("add", &NormalFormFormula::add)
    //     .def("eval", &NormalFormFormula::eval)
    //     .def("avgQ", &NormalFormFormula::avgQ)
    //     .def_property_readonly("ftype", &NormalFormFormula::ftype)
    //     .def_property_readonly("width", &NormalFormFormula::width)
    //     .def_property_readonly("literals", &NormalFormFormula::literals)
    //     ;
}