// https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Utils.h>

#include "jit.hpp"

using namespace llvm;
using namespace llvm::orc;

//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//

// The lexer returns tokens [0-255] if it is an unknown character, otherwise one
// of these for known things.
enum Token
{
    tok_eof = -1,

    // commands
    tok_def = -2,
    tok_extern = -3,

    // primary
    tok_identifier = -4,
    tok_number = -5,

    // control
    tok_if = -6,
    tok_then = -7,
    tok_else = -8,
    tok_for = -9,
    tok_in = -10,

    // operators
    tok_binary = -11,
    tok_unary = -12,
    tok_var = -13,
};

static std::string IdentifierStr; // Filled in if tok_identifier
static double NumVal;             // Filled in if tok_number

/// gettok - Return the next token from standard input.
static int gettok()
{
    static int LastChar = ' ';

    // Skip any whitespace.
    while (isspace(LastChar))
        LastChar = getchar();

    if (isalpha(LastChar))
    { // identifier: [a-zA-Z][a-zA-Z0-9]*
        IdentifierStr = LastChar;
        while (isalnum((LastChar = getchar())))
            IdentifierStr += LastChar;

        if (IdentifierStr == "def")
            return tok_def;
        if (IdentifierStr == "extern")
            return tok_extern;
        if (IdentifierStr == "if")
            return tok_if;
        if (IdentifierStr == "then")
            return tok_then;
        if (IdentifierStr == "else")
            return tok_else;
        if (IdentifierStr == "for")
            return tok_for;
        if (IdentifierStr == "in")
            return tok_in;
        if (IdentifierStr == "binary")
            return tok_binary;
        if (IdentifierStr == "unary")
            return tok_unary;
        if (IdentifierStr == "var")
            return tok_var;

        return tok_identifier;
    }

    if (isdigit(LastChar) || LastChar == '.')
    { // Number: [0-9.]+
        std::string NumStr;
        do
        {
            NumStr += LastChar;
            LastChar = getchar();
        } while (isdigit(LastChar) || LastChar == '.');

        NumVal = strtod(NumStr.c_str(), nullptr);
        return tok_number;
    }

    if (LastChar == '#')
    {
        // Comment until end of line.
        do
            LastChar = getchar();
        while (LastChar != EOF && LastChar != '\n' && LastChar != '\r');

        if (LastChar != EOF)
            return gettok();
    }

    // Check for end of file.    Don't eat the EOF.
    if (LastChar == EOF)
        return tok_eof;

    // Otherwise, just return the character as its ascii value.
    int ThisChar = LastChar;
    LastChar = getchar();
    return ThisChar;
}

//===----------------------------------------------------------------------===//
// Abstract Syntax Tree (aka Parse Tree)
//===----------------------------------------------------------------------===//

namespace
{

    /// ExprAST - Base class for all expression nodes.
    class ExprAST
    {
    public:
        virtual ~ExprAST() = default;

        virtual Value *codegen() = 0;
    };

    /// NumberExprAST - Expression class for numeric literals like "1.0".
    class NumberExprAST : public ExprAST
    {
        double Val;

    public:
        NumberExprAST(double Val) : Val(Val) {}

        Value *codegen() override;
    };

    /// VariableExprAST - Expression class for referencing a variable, like "a".
    class VariableExprAST : public ExprAST
    {
        std::string Name;

    public:
        VariableExprAST(const std::string &Name) : Name(Name) {}

        std::string const & getName() const { return Name; }

        Value *codegen() override;
    };

    /// BinaryExprAST - Expression class for a binary operator.
    class BinaryExprAST : public ExprAST
    {
        char Op;
        std::unique_ptr<ExprAST> LHS, RHS;

    public:
        BinaryExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                      std::unique_ptr<ExprAST> RHS)
            : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}

        Value *codegen() override;
    };

    class UnaryExprAST : public ExprAST
    {
        char Op;
        std::unique_ptr<ExprAST> Operand;

    public:
        UnaryExprAST(char op, std::unique_ptr<ExprAST> operand)
            : Op(op), Operand(std::move(operand))
        { }

        Value * codegen() override;
    };

    /// CallExprAST - Expression class for function calls.
    class CallExprAST : public ExprAST
    {
        std::string Callee;
        std::vector<std::unique_ptr<ExprAST>> Args;

    public:
        CallExprAST(const std::string &Callee,
                    std::vector<std::unique_ptr<ExprAST>> Args)
            : Callee(Callee), Args(std::move(Args)) {}

        Value *codegen() override;
    };

    /// PrototypeAST - This class represents the "prototype" for a function,
    /// which captures its name, and its argument names (thus implicitly the number
    /// of arguments the function takes).
    class PrototypeAST
    {
        std::string Name;
        std::vector<std::string> Args;
        bool isOperator;
        unsigned precedence;

    public:
        PrototypeAST(
            const std::string & Name,
            std::vector<std::string> Args,
            bool isOperator = false,
            unsigned precedence = 0
        )
            : Name(Name)
            , Args(std::move(Args))
            , isOperator(isOperator)
            , precedence(precedence)
        { }

        Function *codegen();

        const std::string &getName() const { return Name; }

        bool isUnaryOp() const { return isOperator && Args.size() == 1; }
        bool isBinaryOp() const { return isOperator && Args.size() == 2; }

        char getOperatorName() const
        {
            assert(isUnaryOp() || isBinaryOp());
            return Name[Name.size() - 1];
        }

        unsigned getBinaryPrecendence() const { return precedence; }
    };

    /// FunctionAST - This class represents a function definition itself.
    class FunctionAST
    {
        std::unique_ptr<PrototypeAST> Proto;
        std::unique_ptr<ExprAST> Body;

    public:
        FunctionAST(std::unique_ptr<PrototypeAST> Proto,
                    std::unique_ptr<ExprAST> Body)
            : Proto(std::move(Proto)), Body(std::move(Body)) {}

        Function *codegen();
    };

    class IfExprAST : public ExprAST
    {
        std::unique_ptr<ExprAST> cond, thenClause, elseClause;

    public:
        IfExprAST(
            std::unique_ptr<ExprAST> cond,
            std::unique_ptr<ExprAST> thenClause,
            std::unique_ptr<ExprAST> elseClause)
            : cond(std::move(cond)), thenClause(std::move(thenClause)), elseClause(std::move(elseClause))
        {
        }

        Value *codegen() override;
    };

    class ForExprAST : public ExprAST
    {
        std::string varName;
        std::unique_ptr<ExprAST> start, end, step, body;

    public:
        ForExprAST(
            std::string const & varName,
            std::unique_ptr<ExprAST> start,
            std::unique_ptr<ExprAST> end,
            std::unique_ptr<ExprAST> step,
            std::unique_ptr<ExprAST> body
        )
            : varName(varName)
            , start(std::move(start))
            , end(std::move(end))
            , step(std::move(step))
            , body(std::move(body))
        { }

        Value * codegen() override;
    };

    class VarExprAST : public ExprAST
    {
        std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> varNames;
        std::unique_ptr<ExprAST> body;

    public:
        VarExprAST(
            std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> varNames,
            std::unique_ptr<ExprAST> body
        )
            : varNames(std::move(varNames))
            , body(std::move(body))
        { }

        Value * codegen() override;
    }

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

/// CurTok/getNextToken - Provide a simple token buffer.    CurTok is the current
/// token the parser is looking at.    getNextToken reads another token from the
/// lexer and updates CurTok with its results.
static int CurTok;
static int getNextToken() { return CurTok = gettok(); }

/// BinopPrecedence - This holds the precedence for each binary operator that is
/// defined.
static std::map<char, int> BinopPrecedence;

/// GetTokPrecedence - Get the precedence of the pending binary operator token.
static int GetTokPrecedence()
{
    if (!isascii(CurTok))
        return -1;

    // Make sure it's a declared binop.
    int TokPrec = BinopPrecedence[CurTok];
    if (TokPrec <= 0)
        return -1;
    return TokPrec;
}

/// LogError* - These are little helper functions for error handling.
std::unique_ptr<ExprAST> LogError(const char *Str)
{
    fprintf(stderr, "Error: %s\n", Str);
    return nullptr;
}

std::unique_ptr<PrototypeAST> LogErrorP(const char *Str)
{
    LogError(Str);
    return nullptr;
}

static std::unique_ptr<ExprAST> ParseExpression();

/// numberexpr ::= number
static std::unique_ptr<ExprAST> ParseNumberExpr()
{
    auto Result = std::make_unique<NumberExprAST>(NumVal);
    getNextToken(); // consume the number
    return std::move(Result);
}

/// parenexpr ::= '(' expression ')'
static std::unique_ptr<ExprAST> ParseParenExpr()
{
    getNextToken(); // eat (.
    auto V = ParseExpression();
    if (!V)
        return nullptr;

    if (CurTok != ')')
        return LogError("expected ')'");
    getNextToken(); // eat ).
    return V;
}

/// identifierexpr
///     ::= identifier
///     ::= identifier '(' expression* ')'
static std::unique_ptr<ExprAST> ParseIdentifierExpr()
{
    std::string IdName = IdentifierStr;

    getNextToken(); // eat identifier.

    if (CurTok != '(') // Simple variable ref.
        return std::make_unique<VariableExprAST>(IdName);

    // Call.
    getNextToken(); // eat (
    std::vector<std::unique_ptr<ExprAST>> Args;
    if (CurTok != ')')
    {
        while (true)
        {
            if (auto Arg = ParseExpression())
                Args.push_back(std::move(Arg));
            else
                return nullptr;

            if (CurTok == ')')
                break;

            if (CurTok != ',')
                return LogError("Expected ')' or ',' in argument list");
            getNextToken();
        }
    }

    // Eat the ')'.
    getNextToken();

    return std::make_unique<CallExprAST>(IdName, std::move(Args));
}

static std::unique_ptr<ExprAST> ParseIfExpr()
{
    getNextToken();

    auto cond = ParseExpression();
    if (!cond)
        return nullptr;

    if (CurTok != tok_then)
        return LogError("expected then");

    auto then = ParseExpression();
    if (!then)
        return nullptr;

    if (CurTok != tok_else)
        return LogError("expected else");

    auto elseClause = ParseExpression();
    if (!elseClause)
        return nullptr;

    return std::make_unique<IfExprAST>(std::move(cond), std::move(then), std::move(elseClause));
}


static std::unique_ptr<ExprAST> ParseForExpr()
{
    getNextToken();

    if (CurTok != tok_identifier)
        return LogError("expected identifier after for");

    std::string name = IdentifierStr;
    getNextToken();

    if (CurTok != '=')
        return LogError("expected '=' after for");
    getNextToken();

    auto start = ParseExpression();
    if (!start)
        return nullptr;
    if (CurTok != ',')
        return LogError("expected ',' after start value");

    auto end = ParseExpression();
    if (!end)
        return nullptr;

    std::unique_ptr<ExprAST> step;
    if (CurTok == ',')
    {
        getNextToken();
        step = ParseExpression();
        if (!step)
            return nullptr;
    }

    if (CurTok != tok_in)
        return LogError("expected 'in' after for");
    getNextToken();

    auto body = ParseExpression();
    if (!body)
        return nullptr;

    return std::make_unique<ForExprAST>(
        name, std::move(start), std::move(end), std::move(step), std::move(body));
}

static std::unique_ptr<ExprAST> ParseVarExpr()
{
    getNextToken();

    std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> varNames;

    while (true)
    {
        if (CurTok != tok_identifier)
            return LogError("expected identifier after var");

        std::string name = IdentifierStr;
        getNextToken();

        std::unique_ptr<ExprAST> init;
        if (CurTok == '=')
        {
            getNextToken();

            init = ParseExpression();
            if (!init)
                return nullptr;
        }

        varNames.push_back(std::make_pair(name, std::move(init)));

        if (CurTok != ',') break;
        getNextToken();
    }

    if (CurTok != tok_in)
        return LogError("expected 'in' keyword after 'var'");
    getNextToken();

    auto body = ParseExpression();
    if (!body)
        return nullptr;

    return std::make_unique<VarExprAST>(std::move(varNames), std::move(body));
}

/// primary
///     ::= identifierexpr
///     ::= numberexpr
///     ::= parenexpr
static std::unique_ptr<ExprAST> ParsePrimary()
{
    switch (CurTok)
    {
    case tok_identifier:
        return ParseIdentifierExpr();

    case tok_number:
        return ParseNumberExpr();

    case '(':
        return ParseParenExpr();

    case tok_if:
        return ParseIfExpr();

    case tok_for:
        return ParseForExpr();

    case tok_var:
        return ParseVarExpr();

    default:
        return LogError("unknown token when expecting an expression");
    }
}

static std::unique_ptr<ExprAST> ParseUnary()
{
    if (!isascii(CurTok) || CurTok == '(' || CurTok == ',')
        return ParsePrimary();

    int OpCode = CurTok;
    if (auto operand = ParseUnary())
        return std::make_unique<UnaryExprAST>(OpCode, std::move(operand));
    return nullptr;
}

/// binoprhs
///     ::= ('+' primary)*
static std::unique_ptr<ExprAST> ParseBinOpRHS(int ExprPrec, std::unique_ptr<ExprAST> LHS)
{
    // If this is a binop, find its precedence.
    while (true)
    {
        int TokPrec = GetTokPrecedence();

        // If this is a binop that binds at least as tightly as the current binop,
        // consume it, otherwise we are done.
        if (TokPrec < ExprPrec)
            return LHS;

        // Okay, we know this is a binop.
        int BinOp = CurTok;
        getNextToken(); // eat binop

        auto RHS = ParseUnary();
        if (!RHS)
            return nullptr;

        // If BinOp binds less tightly with RHS than the operator after RHS, let
        // the pending operator take RHS as its LHS.
        int NextPrec = GetTokPrecedence();
        if (TokPrec < NextPrec)
        {
            RHS = ParseBinOpRHS(TokPrec + 1, std::move(RHS));
            if (!RHS)
                return nullptr;
        }

        // Merge LHS/RHS.
        LHS =
            std::make_unique<BinaryExprAST>(BinOp, std::move(LHS), std::move(RHS));
    }
}

/// expression
///     ::= primary binoprhs
///
static std::unique_ptr<ExprAST> ParseExpression()
{
    auto LHS = ParseUnary();
    if (!LHS)
        return nullptr;

    return ParseBinOpRHS(0, std::move(LHS));
}

/// prototype
///     ::= id '(' id* ')'
static std::unique_ptr<PrototypeAST> ParsePrototype()
{
    std::string name;

    unsigned kind = 0;  // 0 = identifier, 1 = unary, 2 = binary
    unsigned binaryPrecedence = 30;

    switch (CurTok)
    {
    case tok_identifier:
    {
        name = IdentifierStr;
        kind = 0;
        getNextToken();
    } break;

    case tok_binary:
    {
        getNextToken();
        if (!isascii(CurTok))
            return LogErrorP("Expected binary operator");
        name = "binary";
        name += (char)CurTok;
        kind = 2;
        getNextToken();

        if (CurTok == tok_number)
        {
            if (NumVal < 1 || NumVal > 100)
                return LogErrorP("Invalid precedence; must be 1..100");
            binaryPrecedence = (unsigned)NumVal;
            getNextToken();
        }
    } break;

    case tok_unary:
    {
        getNextToken();
        if (!isascii(CurTok))
            return LogErrorP("Expected unary operator");
        name = "unary";
        name += (char)CurTok;
        kind = 1;
        getNextToken();
    } break;

    default:
        return nullptr;
    }

    if (CurTok != '(')
        return LogErrorP("Expected '(' in prototype");

    std::vector<std::string> ArgNames;
    while (getNextToken() == tok_identifier)
        ArgNames.push_back(IdentifierStr);
    if (CurTok != ')')
        return LogErrorP("Expected ')' in prototype");

    // success.
    getNextToken(); // eat ')'.

    if (kind && ArgNames.size() != kind)
        return LogErrorP("Invalid number of operands for operator");

    return std::make_unique<PrototypeAST>(name, std::move(ArgNames), kind != 0, binaryPrecedence);
}

/// definition ::= 'def' prototype expression
static std::unique_ptr<FunctionAST> ParseDefinition()
{
    getNextToken(); // eat def.
    auto Proto = ParsePrototype();
    if (!Proto)
        return nullptr;

    if (auto E = ParseExpression())
        return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
    return nullptr;
}

/// toplevelexpr ::= expression
static std::unique_ptr<FunctionAST> ParseTopLevelExpr()
{
    if (auto E = ParseExpression())
    {
        // Make an anonymous proto.
        auto Proto = std::make_unique<PrototypeAST>("__anon_expr",
                                                    std::vector<std::string>());
        return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
    }
    return nullptr;
}

/// external ::= 'extern' prototype
static std::unique_ptr<PrototypeAST> ParseExtern()
{
    getNextToken(); // eat extern.
    return ParsePrototype();
}

//===----------------------------------------------------------------------===//
// Code Generation
//===----------------------------------------------------------------------===//

static LLVMContext TheContext;
static IRBuilder<> Builder(TheContext);
static std::unique_ptr<Module> TheModule;
static std::map<std::string, AllocaInst *> NamedValues;
static std::unique_ptr<legacy::FunctionPassManager> TheFPM;
static std::unique_ptr<KaleidoscopeJIT> TheJIT;
static std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionProtos;

Value *LogErrorV(const char *Str)
{
    LogError(Str);
    return nullptr;
}

Function *getFunction(std::string Name)
{
    // First, see if the function has already been added to the current module.
    if (auto *F = TheModule->getFunction(Name))
        return F;

    // If not, check whether we can codegen the declaration from some existing
    // prototype.
    auto FI = FunctionProtos.find(Name);
    if (FI != FunctionProtos.end())
        return FI->second->codegen();

    // If no existing prototype exists, return null.
    return nullptr;
}

static AllocaInst * CreateEntryBlockAlloca(Function * TheFunction, std::string const varName)
{
    IRBuilder<> TmpB(&TheFunction->getEntryBlock(), TheFunction->getEntryBlock().begin());
    return TmpB.CreateAlloca(Type::getDoubleTy(TheContext), 0, varName.c_str());
}

static AllocaInst * CreateEntryBlockAlloca(Function * TheFunction, StringRef const varName)
{
    IRBuilder<> TmpB(&TheFunction->getEntryBlock(), TheFunction->getEntryBlock().begin());
    return TmpB.CreateAlloca(Type::getDoubleTy(TheContext), 0, varName);
}

Value *NumberExprAST::codegen()
{
    return ConstantFP::get(TheContext, APFloat(Val));
}

Value *VariableExprAST::codegen()
{
    // Look this variable up in the function.
    Value *V = NamedValues[Name];
    if (!V)
        return LogErrorV("Unknown variable name");

    return Builder.CreateLoad(V, Name.c_str());
}

Value *BinaryExprAST::codegen()
{
    if (Op == '=')
    {
        auto * lhse = dynamic_cast<VariableExprAST*>(LHS.get());
        if (!lhse)
            return LogErrorV("destination of '=' must be a variable");

        auto * val = RHS->codegen();
        if (!val)
            return nullptr;

        auto * variable = NamedValues[lhse->getName()];
        if (!variable)
            return LogErrorV("Unknown variable name");

        Builder.CreateStore(val, variable);
        return val;
    }

    Value *L = LHS->codegen();
    Value *R = RHS->codegen();
    if (!L || !R)
        return nullptr;

    switch (Op)
    {
    case '+':
        return Builder.CreateFAdd(L, R, "addtmp");
    case '-':
        return Builder.CreateFSub(L, R, "subtmp");
    case '*':
        return Builder.CreateFMul(L, R, "multmp");
    case '<':
        L = Builder.CreateFCmpULT(L, R, "cmptmp");
        // Convert bool 0/1 to double 0.0 or 1.0
        return Builder.CreateUIToFP(L, Type::getDoubleTy(TheContext), "booltmp");

    default:
        break;
    }

    auto * f = getFunction(std::string("binary") + Op);
    assert(f);

    Value * Ops[2] = { L, R };
    return Builder.CreateCall(f, Ops, "binop");
}

Value * UnaryExprAST::codegen()
{
    auto * operandV = Operand->codegen();
    if (!operandV)
        return nullptr;

    auto * f = getFunction(std::string("unary") + Op);
    if (!f)
        return LogErrorV("Unknown unary operator");
    return Builder.CreateCall(f, operandV, "unop");
}

Value *CallExprAST::codegen()
{
    // Look up the name in the global module table.
    Function *CalleeF = getFunction(Callee);
    if (!CalleeF)
        return LogErrorV("Unknown function referenced");

    // If argument mismatch error.
    if (CalleeF->arg_size() != Args.size())
        return LogErrorV("Incorrect # arguments passed");

    std::vector<Value *> ArgsV;
    for (unsigned i = 0, e = Args.size(); i != e; ++i)
    {
        ArgsV.push_back(Args[i]->codegen());
        if (!ArgsV.back())
            return nullptr;
    }

    return Builder.CreateCall(CalleeF, ArgsV, "calltmp");
}


Value * IfExprAST::codegen()
{
    auto * condV = cond->codegen();
    if (!condV)
        return nullptr;

    condV = Builder.CreateFCmpONE(condV, ConstantFP::get(TheContext, APFloat(0.0)), "ifcond");

    auto * theFunction = Builder.GetInsertBlock()->getParent();

    auto * thenBlock = BasicBlock::Create(TheContext, "then", theFunction);
    auto * elseBlock = BasicBlock::Create(TheContext, "else");
    auto * mergeBlock = BasicBlock::Create(TheContext, "ifcont");

    Builder.CreateCondBr(condV, thenBlock, elseBlock);

    Builder.SetInsertPoint(thenBlock);

    auto * thenV = thenClause->codegen();
    if (!thenV)
        return nullptr;

    Builder.CreateBr(mergeBlock);
    thenBlock = Builder.GetInsertBlock();

    theFunction->getBasicBlockList().push_back(elseBlock);
    Builder.SetInsertPoint(elseBlock);

    auto * elseV = elseClause->codegen();
    if (!elseV)
        return nullptr;

    Builder.CreateBr(mergeBlock);
    elseBlock = Builder.GetInsertBlock();

    theFunction->getBasicBlockList().push_back(mergeBlock);
    Builder.SetInsertPoint(mergeBlock);
    auto * pn = Builder.CreatePHI(Type::getDoubleTy(TheContext), 2, "iftmp");

    pn->addIncoming(thenV, thenBlock);
    pn->addIncoming(elseV, elseBlock);
    return pn;
}


Value * ForExprAST::codegen()
{
    auto * theFunction = Builder.GetInsertBlock()->getParent();

    auto * Alloca = CreateEntryBlockAlloca(theFunction, varName);

    auto * startVal = start->codegen();
    if (!startVal)
        return nullptr;

    Builder.CreateStore(startVal, Alloca);

    auto * loopBlock = BasicBlock::Create(TheContext, "loop", theFunction);
    Builder.CreateBr(loopBlock);
    Builder.SetInsertPoint(loopBlock);

    auto * oldVal = NamedValues[varName];
    NamedValues.insert(std::make_pair(varName, Alloca));

    if (!body->codegen())
        return nullptr;

    Value * stepVal = nullptr;
    if (step)
    {
        stepVal = step->codegen();
        if (!stepVal)
            return nullptr;
        else
            stepVal = ConstantFP::get(TheContext, APFloat(1.0));
    }

    auto * endCond = end->codegen();
    if (!endCond)
        return nullptr;

    auto * curVal = Builder.CreateLoad(Alloca, varName.c_str());
    auto * nextVal = Builder.CreateFAdd(curVal, stepVal, "nextvar");
    Builder.CreateStore(nextVal, Alloca);

    endCond = Builder.CreateFCmpONE(endCond, ConstantFP::get(TheContext, APFloat(0.0)), "loopcond");

    auto * afterBlock = BasicBlock::Create(TheContext, "afterloop", theFunction);

    Builder.CreateCondBr(endCond, loopBlock, afterBlock);
    Builder.SetInsertPoint(afterBlock);

    if (oldVal)
        NamedValues[varName] = oldVal;
    else
        NamedValues.erase(varName);

    return Constant::getNullValue(Type::getDoubleTy(TheContext));
}

Value * VarExprAST::codegen()
{
    std::vector<AllocaInst *> oldBindings;

    auto * theFunction = Builder.GetInsertBlock()->getParent();

    for (unsigned i = 0, e = varNames.size(); i != e; ++i)
    {
        const std::string & varName = varNames[i].first;
        auto * init = varNames[i].second.get();
        Value * initVal;
        if (init)
        {
            initVal = init->codegen();
            if (!initVal)
                return nullptr;
        }
        else
        {
            initVal = ConstantFP::get(TheContext, APFloat(0.0));
        }

        auto * Alloca = CreateEntryBlockAlloca(theFunction, varName);
        Builder.CreateStore(initVal, Alloca);

        oldBindings.push_back(NamedValues[varName]);

        NamedValues.insert(std::make_pair(varName, Alloca));
    }

    auto * bodyVal = body->codegen();
    if (!bodyVal)
        return nullptr;

    for (unsigned i = 0, e = varNames.size(); i != e; ++i)
        NamedValues[varNames[i].first] = oldBindings[i];

    return bodyVal;
}

Function *PrototypeAST::codegen()
{
    // Make the function type:    double(double,double) etc.
    std::vector<Type *> Doubles(Args.size(), Type::getDoubleTy(TheContext));
    FunctionType *FT =
        FunctionType::get(Type::getDoubleTy(TheContext), Doubles, false);

    Function *F =
        Function::Create(FT, Function::ExternalLinkage, Name, TheModule.get());

    // Set names for all arguments.
    unsigned Idx = 0;
    for (auto &Arg : F->args())
        Arg.setName(Args[Idx++]);

    return F;
}

Function * FunctionAST::codegen()
{
    // Transfer ownership of the prototype to the FunctionProtos map, but keep a
    // reference to it for use below.
    auto & P = *Proto;
    FunctionProtos[Proto->getName()] = std::move(Proto);
    auto * TheFunction = getFunction(P.getName());
    if (!TheFunction)
        return nullptr;

    if (P.isBinaryOp())
        BinopPrecedence.insert(std::make_pair(P.getOperatorName(), P.getBinaryPrecendence()));

    // Create a new basic block to start insertion into.
    auto * BB = BasicBlock::Create(TheContext, "entry", TheFunction);
    Builder.SetInsertPoint(BB);

    // Record the function arguments in the NamedValues map.
    NamedValues.clear();
    for (auto &Arg : TheFunction->args())
    {
        auto * Alloca = CreateEntryBlockAlloca(TheFunction, Arg.getName());
        Builder.CreateStore(&Arg, Alloca);
        NamedValues.insert(std::make_pair(Arg.getName(), Alloca));
    }

    if (Value *RetVal = Body->codegen())
    {
        // Finish off the function.
        Builder.CreateRet(RetVal);

        // Validate the generated code, checking for consistency.
        verifyFunction(*TheFunction);

        // Run the optimizer on the function.
        TheFPM->run(*TheFunction);

        return TheFunction;
    }

    // Error reading body, remove function.
    TheFunction->eraseFromParent();
    return nullptr;
}

//===----------------------------------------------------------------------===//
// Top-Level parsing and JIT Driver
//===----------------------------------------------------------------------===//

static void InitializeModuleAndPassManager()
{
    // Open a new module.
    TheModule = std::make_unique<Module>("my cool jit", TheContext);
    TheModule->setDataLayout(TheJIT->getTargetMachine().createDataLayout());

    // Create a new pass manager attached to it.
    TheFPM = std::make_unique<legacy::FunctionPassManager>(TheModule.get());

    // Do simple "peephole" optimizations and bit-twiddling optzns.
    TheFPM->add(createInstructionCombiningPass());
    // Reassociate expressions.
    TheFPM->add(createReassociatePass());
    // Eliminate Common SubExpressions.
    TheFPM->add(createGVNPass());
    // Simplify the control flow graph (deleting unreachable blocks, etc).
    TheFPM->add(createCFGSimplificationPass());

    TheFPM->add(createPromoteMemoryToRegisterPass());
    TheFPM->add(createInstructionCombiningPass());
    TheFPM->add(createReassociatePass());

    TheFPM->doInitialization();
}

static void HandleDefinition()
{
    if (auto FnAST = ParseDefinition())
    {
        if (auto *FnIR = FnAST->codegen())
        {
            fprintf(stderr, "Read function definition:\n");
            FnIR->print(errs());
            fprintf(stderr, "\n");
            TheJIT->addModule(std::move(TheModule));
            InitializeModuleAndPassManager();
        }
    }
    else
    {
        // Skip token for error recovery.
        getNextToken();
    }
}

static void HandleExtern()
{
    if (auto ProtoAST = ParseExtern())
    {
        if (auto *FnIR = ProtoAST->codegen())
        {
            fprintf(stderr, "Read extern: \n");
            FnIR->print(errs());
            fprintf(stderr, "\n");
            FunctionProtos[ProtoAST->getName()] = std::move(ProtoAST);
        }
    }
    else
    {
        // Skip token for error recovery.
        getNextToken();
    }
}

static void HandleTopLevelExpression()
{
    // Evaluate a top-level expression into an anonymous function.
    if (auto FnAST = ParseTopLevelExpr())
    {
        if (FnAST->codegen())
        {
            // JIT the module containing the anonymous expression, keeping a handle so
            // we can free it later.
            auto H = TheJIT->addModule(std::move(TheModule));
            InitializeModuleAndPassManager();

            // Search the JIT for the __anon_expr symbol.
            auto ExprSymbol = TheJIT->findSymbol("__anon_expr");
            assert(ExprSymbol && "Function not found");

            // Get the symbol's address and cast it to the right type (takes no
            // arguments, returns a double) so we can call it as a native function.
            double (*FP)() = (double (*)())(intptr_t)cantFail(ExprSymbol.getAddress());
            fprintf(stderr, "Evaluated to %f\n", FP());

            // Delete the anonymous expression module from the JIT.
            TheJIT->removeModule(H);
        }
    }
    else
    {
        // Skip token for error recovery.
        getNextToken();
    }
}

/// top ::= definition | external | expression | ';'
static void MainLoop()
{
    while (true)
    {
        fprintf(stderr, "ready> ");
        switch (CurTok)
        {
        case tok_eof:
            return;
        case ';': // ignore top-level semicolons.
            getNextToken();
            break;
        case tok_def:
            HandleDefinition();
            break;
        case tok_extern:
            HandleExtern();
            break;
        default:
            HandleTopLevelExpression();
            break;
        }
    }
}

//===----------------------------------------------------------------------===//
// "Library" functions that can be "extern'd" from user code.
//===----------------------------------------------------------------------===//

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

/// putchard - putchar that takes a double and returns 0.
extern "C" DLLEXPORT double putchard(double X)
{
    fputc((char)X, stderr);
    return 0;
}

/// printd - printf that takes a double prints it as "%f\n", returning 0.
extern "C" DLLEXPORT double printd(double X)
{
    fprintf(stderr, "%f\n", X);
    return 0;
}

//===----------------------------------------------------------------------===//
// Main driver code.
//===----------------------------------------------------------------------===//

int main()
{
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();
    InitializeNativeTargetAsmParser();

    // Install standard binary operators.
    // 1 is lowest precedence.
    BinopPrecedence['='] = 10;
    BinopPrecedence['<'] = 10;
    BinopPrecedence['+'] = 20;
    BinopPrecedence['-'] = 20;
    BinopPrecedence['*'] = 40; // highest.

    // Prime the first token.
    fprintf(stderr, "ready> ");
    getNextToken();

    TheJIT = std::make_unique<KaleidoscopeJIT>();

    InitializeModuleAndPassManager();

    // Run the main "interpreter loop" now.
    MainLoop();

    return 0;
}