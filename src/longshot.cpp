#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(longshot, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers",
        py::arg("i") = 1, py::arg("j") = 2);
}

// #define __INJECT_DEBUGGER__

// #include "BooleanFunction.hpp"
// #include "LinearThreshold.hpp"
// #include "SparseSet.hpp"
// #include "TruthTable.hpp"
// #include "DecisionTree.hpp"

// #include "search_tree.hpp"

// using namespace longshot;

// namespace py = pybind11;

// class PyBooleanFunction : public BooleanFunction
// {
// public:
//     using BooleanFunction::BooleanFunction;

//     bool operator () (const bits_t & x) const override {
//         PYBIND11_OVERRIDE_PURE_NAME(
//             bool,
//             BooleanFunction,
//             "__call__",
//             operator(),
//             x
//         );
//     }

//     bool is_constant() const  override {
//         PYBIND11_OVERRIDE(
//             bool,
//             BooleanFunction,
//             is_constant,
//         );
//     }
// };

// class PySparseSet : public SparseSet
// {
// public:
//     using SparseSet::SparseSet;

//     bool operator () (const bits_t & x) const override {
//         PYBIND11_OVERRIDE_NAME(
//             bool,
//             SparseSet,
//             "__call__",
//             operator(),
//             x
//         );
//     }

//     bool is_constant() const override {
//         PYBIND11_OVERRIDE(
//             bool,
//             SparseSet,
//             is_constant,
//         );
//     }
// };

// class PyLinearThreshold : public LinearThreshold
// {
// public:
//     using LinearThreshold::LinearThreshold;

//     bool operator () (const bits_t & x) const override {
//         PYBIND11_OVERRIDE_NAME(
//             bool,
//             LinearThreshold,
//             "__call__",
//             operator(),
//             x
//         );
//     }

//     bool is_constant() const override {
//         PYBIND11_OVERRIDE(
//             bool,
//             LinearThreshold,
//             is_constant,
//         );
//     }
// };

// class PyTruthTable : public TruthTable
// {
// public:
//     using TruthTable::TruthTable;

//     bool operator () (const bits_t & x) const override {
//         PYBIND11_OVERRIDE_NAME(
//             bool,
//             TruthTable,
//             "__call__",
//             operator(),
//             x
//         );
//     }

//     bool is_constant() const override {
//         PYBIND11_OVERRIDE(
//             bool,
//             TruthTable,
//             is_constant,
//         );
//     }
// };

// PYBIND11_MODULE(longshot, m) {
//     py::module_ tm = m.def_submodule("tree");
//     py::module_ fm = m.def_submodule("function");
//     py::module_ am = m.def_submodule("algorithm");

//     py::class_<BooleanFunction, PyBooleanFunction>(fm, 
//         "BooleanFunction")
//         .def(py::init<int>())
//         .def_readonly("num_var", & BooleanFunction::num_var)
//         ;
//     py::class_<SparseSet, BooleanFunction, PySparseSet>(fm, 
//         "SparseSet")
//         .def(py::init<int>())
//         .def(py::init<int, SparseSet::bks_t>())
//         .def("__call__", & SparseSet::operator())
//         .def(py::self == py::self)
//         .def("insert_black", & SparseSet::insert_black)
//         .def("erase_black", & SparseSet::erase_black)
//         .def("sub_function", & SparseSet::sub_function)
//         .def_property_readonly("is_constant", & SparseSet::is_constant)
//         .def_property_readonly("num_blacks", & SparseSet::get_num_blacks)
//         .def_property_readonly("blacks", & SparseSet::get_blacks)
//         ;
//     py::class_<LinearThreshold, BooleanFunction, PyLinearThreshold>(fm, 
//         "LinearThreshold")
//         .def(py::init<double, const LinearThreshold::coeff_t &>())
//         .def("__call__", & LinearThreshold::operator())
//         .def(py::self == py::self)
//         .def("sub_function", & LinearThreshold::sub_function)
//         .def_property_readonly("is_constant", & LinearThreshold::is_constant)
//         .def_property_readonly("coeff", & LinearThreshold::get_coeff)
//         .def_property_readonly("bias", & LinearThreshold::get_bias)
//         ;
        
//     py::class_<BooleanFun>(fm, 
//         "_Boolean_Fun_")
//         .def(py::init<int>())
//         .def(py::init<int, const std::string &>())
//         .def(py::init<const BooleanFun &>())
//         .def(py::self < py::self)
//         .def(py::self == py::self)
//         .def("var_num", &BooleanFun::var_num)
//         .def("set_anf", &BooleanFun::set_anf)
//         .def("set_coe_list", &BooleanFun::set_coe_list)
//         .def("set_anf_coe", &BooleanFun::set_anf_coe)
//         .def("set_anf_coe_done", &BooleanFun::set_anf_coe_done)
//         .def("set_truth_table", &BooleanFun::set_truth_table)
//         .def("set_truth_table_orbit", &BooleanFun::set_truth_table_orbit)
//         .def("set_truth_table_done", &BooleanFun::set_truth_table_done)
//         .def("set_truth_table_hex", &BooleanFun::set_truth_table_hex)
//         .def("set_truth_table_random", &BooleanFun::set_truth_table_random)
//         .def("set_random_sym", &BooleanFun::set_random_sym)
//         .def("sub_function", py::overload_cast<int>(&BooleanFun::sub_function, py::const_))
//         .def("sub_function", py::overload_cast<int, int>(&BooleanFun::sub_function, py::const_))
//         .def("restriction", &BooleanFun::restriction)
//         // NOT support BooleanFun::value(int, ...) method
//         .def("value_dec", &BooleanFun::value_dec)
//         .def("get_anf", &BooleanFun::get_anf)
//         .def("get_coe_list", &BooleanFun::get_coe_list)
//         .def("get_truth_table_hex", &BooleanFun::get_truth_table_hex)
//         .def("get_anf_coe", &BooleanFun::get_anf_coe)
//         .def("get_degree", &BooleanFun::get_degree)
//         .def("is_equal", &BooleanFun::is_equal)
//         .def("negate", &BooleanFun::negate)
//         .def("add", &BooleanFun::add)
//         .def("mult", &BooleanFun::mult)
//         // NOT support BooleanFunc::apply_affine_trans(const AffineTrans&) method
//         .def("trim_degree_below", &BooleanFun::trim_degree_below)
//         .def("dist", &BooleanFun::dist)
//         .def("is_homogenous", &BooleanFun::is_homogenous)
//         .def("walsh_transform", &BooleanFun::walsh_transform)
//         .def("cost", &BooleanFun::cost)
//         .def("nonlinearity", py::overload_cast<>(&BooleanFun::nonlinearity, py::const_))
//         .def("nonlinearity", py::overload_cast<int>(&BooleanFun::nonlinearity, py::const_))
//         // NOT support get_truth_table_ptr() method
//         // NOT support get_anf_ptr() method
//         // NOT support truth_table_to_univariate(Field&) method
//         // NOT support truth_table_to_univariate(Field&) method
//         // NOT support is_univariate_boolean(Field&) method
//         // NOT support is_univariate_boolean(Field&) method
//         // NOT support univariate_to_truth_table(Field&) method
//         // NOT support univariate_to_truth_table(Field&) method
//         // NOT support set_trace_univariate(const std::string&, Field&) method
//         // NOT support set_trace_univariate(const std::string&, Field&) method
//         // NOT support get_un_ptr() method
//         .def("is_monotone", &BooleanFun::is_monotone)
//         .def("inf", &BooleanFun::inf)
//         ;

//     py::class_<TruthTable, BooleanFun, BooleanFunction, PyTruthTable>(fm, 
//         "TruthTable", py::multiple_inheritance())
//         .def(py::init<int>())
//         .def(py::init<int, const std::string &>())
//         .def(py::init<const BooleanFun &>())
//         .def("__call__", &TruthTable::operator())
//         ;

//     py::class_<DecisionTree>(tm, 
//         "DecisionTree")
//         .def(py::init<>())
//         .def(py::init<bool>())
//         .def(py::init<int, DecisionTree &, DecisionTree &>())
//         .def(py::init<const DecisionTree &>())
//         .def(py::self >> py::self)
//         .def(py::self >> bool())
//         .def(py::self << py::self)
//         .def(bool() << py::self)
//         .def("__truediv__", [](DecisionTree & lhs, DecisionTree & rhs){
//             return lhs / rhs;
//         }, py::is_operator())
//         .def("__itruediv__", [](DecisionTree & lhs, DecisionTree & rhs){
//             lhs /= rhs;
//             return lhs; // Python __itruedive__ method need to return an object
//         }, py::is_operator())
//         .def("__str__", &DecisionTree::toString)
//         .def_property_readonly("parent", &DecisionTree::parent)
//         .def_property_readonly("left", &DecisionTree::left)
//         .def_property_readonly("right", &DecisionTree::right)
//         .def_property_readonly("label", &DecisionTree::get_label)
//         .def_property_readonly("value", &DecisionTree::get_value)
//         ;
        
//     tm.def("internode_constructor", &DecisionTree::internode_constructor);

//     // These definition order MUST NOT be changed. 
//     am.def("avg_optimal_tree", py::overload_cast<const LinearThreshold &>(&avg_optimal_tree));
//     am.def("avg_optimal_tree", py::overload_cast<const BooleanFunction &>(&avg_optimal_tree));
// }