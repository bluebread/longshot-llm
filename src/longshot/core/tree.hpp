#ifndef __LONGSHOT_CORE_TREE_HPP__
#define __LONGSHOT_CORE_TREE_HPP__

#include <stdint.h>
#include <stdexcept>

namespace longshot
{
    class TreeNode
    {
    public:
        int var; // if leaf, this represents the value on input
        TreeNode *left; // 0-edge
        TreeNode *right; // 1-edge

        TreeNode() : var(), left(nullptr), right(nullptr) {}
        TreeNode(int v) : var(v), left(nullptr), right(nullptr) {}
        TreeNode(int v, TreeNode *l, TreeNode *r) : var(v), left(l), right(r) {}

        bool is_leaf() const
        {
            return left == nullptr && right == nullptr;
        }
    };

    class DecisionTree
    {
    private:
        TreeNode *root;

    public:
        DecisionTree() : root(nullptr) {} // empty tree
        DecisionTree(int v) : root(new TreeNode(v)) {} // constant tree
        DecisionTree(int v, const DecisionTree & l, const DecisionTree & r) 
        {
            root = new TreeNode(v, l.root, r.root);
        }
        DecisionTree(TreeNode *node) : root(node) {}
        DecisionTree(const DecisionTree &other) : root(other.root) {}

        ~DecisionTree() {}

    private:
        void _delete_tree_core(TreeNode *node)
        {
            if (node == nullptr)
                return;

            _delete_tree_core(node->left);
            _delete_tree_core(node->right);
            delete node;
        }
    public:
        void delete_tree() {
            _delete_tree_core(root);
            root = nullptr;
        }

        DecisionTree ltree() const {
            return DecisionTree(root->left);
        }
        DecisionTree rtree() const {
            return DecisionTree(root->right);
        }

        bool is_constant() const {
            if (root == nullptr)
                throw std::runtime_error("Tree has been deleted.");
            return root->is_leaf();
        }
        int var() const {
            if (root == nullptr)
                throw std::runtime_error("Tree has been deleted.");
            return root->var;
        }

        bool decide(uint32_t x) const {
            if (root == nullptr)
                throw std::runtime_error("Tree has been deleted.");
            if (root->is_leaf())
                return root->var;
            if (x & (1 << root->var))
                return rtree().decide(x);
            else
                return ltree().decide(x);            
        }
    };
}

#endif