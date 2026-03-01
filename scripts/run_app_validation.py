#!/usr/bin/env python3
"""
LangGraph Configuration Validation Script

Purpose:
- Validate the LangGraph implementation in src/ directory
- Identify issues preventing src/ module from being importable
- Demonstrate correct initialization patterns (factory pattern vs module-level)
- Serve as reference for migrating run_full_evaluation.py to use src/ modules

This script validates:
1. Environment configuration (API keys, Qdrant connectivity)
2. Module imports (which src/ modules are broken)
3. Retriever factory pattern (correct way to create retrievers)
4. Graph compilation (all 4 LangGraph workflows)
5. Functional execution (test queries through each graph)
6. Configuration consistency (comparing src/ vs run_full_evaluation.py)

Exit codes:
- 0: All validations passed
- 1: One or more validations failed
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import traceback

# Add project root to Python path to enable src/ imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Color codes for terminal output
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}\n")


def print_check(name: str, passed: bool, details: str = ""):
    """Print a check result with color coding"""
    symbol = f"{Colors.GREEN}✓{Colors.END}" if passed else f"{Colors.RED}✗{Colors.END}"
    status = (
        f"{Colors.GREEN}PASS{Colors.END}" if passed else f"{Colors.RED}FAIL{Colors.END}"
    )
    print(f"{symbol} {name:<50} [{status}]")
    if details and not passed:
        print(f"  {Colors.YELLOW}→ {details}{Colors.END}")


def print_info(message: str):
    """Print an informational message"""
    print(f"{Colors.BLUE}ℹ {message}{Colors.END}")


def print_warning(message: str):
    """Print a warning message"""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")


def print_error(message: str):
    """Print an error message"""
    print(f"{Colors.RED}✗ {message}{Colors.END}")


# ==============================================================================
# 1. ENVIRONMENT VALIDATION
# ==============================================================================


def check_environment() -> Dict[str, bool]:
    """Validate environment configuration"""
    print_section("1. ENVIRONMENT VALIDATION")

    results = {}

    # Check API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    results["openai_key"] = bool(openai_key)
    print_check(
        "OPENAI_API_KEY set",
        results["openai_key"],
        "Required for LLM and embeddings" if not results["openai_key"] else "",
    )

    cohere_key = os.getenv("COHERE_API_KEY")
    results["cohere_key"] = bool(cohere_key)
    print_check(
        "COHERE_API_KEY set",
        results["cohere_key"],
        "Required for rerank retriever (optional for other retrievers)",
    )

    # Check Qdrant connectivity
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(host="localhost", port=6333, timeout=5)
        collections = client.get_collections()
        results["qdrant"] = True
        print_check(
            "Qdrant at localhost:6333",
            True,
            f"Found {len(collections.collections)} collections",
        )
    except Exception as e:
        results["qdrant"] = False
        print_check("Qdrant at localhost:6333", False, str(e))

    # Check critical Python imports
    critical_imports = [
        ("langchain", "LangChain core"),
        ("langchain_openai", "LangChain OpenAI integration"),
        ("langgraph", "LangGraph"),
        ("qdrant_client", "Qdrant client"),
        ("datasets", "HuggingFace datasets"),
        ("ragas", "RAGAS evaluation"),
    ]

    for module_name, description in critical_imports:
        try:
            __import__(module_name)
            results[f"import_{module_name}"] = True
            print_check(f"Import {module_name}", True)
        except ImportError as e:
            results[f"import_{module_name}"] = False
            print_check(f"Import {module_name}", False, str(e))

    return results


# ==============================================================================
# 2. MODULE IMPORT VALIDATION
# ==============================================================================


def test_module_imports() -> Dict[str, Dict[str, Any]]:
    """Test importing each src/ module individually"""
    print_section("2. MODULE IMPORT VALIDATION")

    results = {}
    modules_to_test = [
        ("src.config", "Configuration (LLM, embeddings, Qdrant)"),
        ("src.state", "State schema (TypedDict)"),
        ("src.prompts", "Prompt templates"),
        ("src.retrievers", "Retriever instances (EXPECTED TO FAIL)"),
        ("src.graph", "LangGraph workflows"),
        ("src.utils", "Utility functions"),
    ]

    for module_name, description in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[""])
            results[module_name] = {"success": True, "module": module, "error": None}
            print_check(f"{module_name:<25} - {description}", True)
        except Exception as e:
            results[module_name] = {
                "success": False,
                "module": None,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            print_check(f"{module_name:<25} - {description}", False, str(e))

    # Report details for failed imports
    failed_imports = [name for name, result in results.items() if not result["success"]]
    if failed_imports:
        print(f"\n{Colors.YELLOW}Failed Import Details:{Colors.END}")
        for module_name in failed_imports:
            print(f"\n{Colors.RED}{module_name}:{Colors.END}")
            print(results[module_name]["traceback"])

    return results


# ==============================================================================
# 3. CORRECT INITIALIZATION PATTERN DEMONSTRATION
# ==============================================================================


def demonstrate_correct_pattern() -> Optional[Dict[str, Any]]:
    """Demonstrate correct retriever factory pattern using actual src/ modules"""
    print_section("3. VALIDATION OF src/ MODULE IMPLEMENTATION")

    try:
        # Use actual src/ modules (not inline code)
        from src.utils import load_documents_from_huggingface
        from src.config import create_vector_store
        from src.retrievers import create_retrievers

        print_info("Loading documents using src.utils...")

        # Step 1: Load source documents
        documents = load_documents_from_huggingface()
        print_check(
            "Load documents from HuggingFace",
            True,
            f"Loaded {len(documents)} documents",
        )

        # Step 2: Create vector store using src.config factory
        print_info("Creating vector store using src.config...")
        vector_store = create_vector_store(
            documents, collection_name="gdelt_validation_test", recreate_collection=True
        )
        print_check(
            "Create vector store via factory", True, "Collection created and populated"
        )

        # Step 3: Create retrievers using src.retrievers factory
        print_info("Creating retrievers using src.retrievers...")
        retrievers = create_retrievers(documents, vector_store)
        print_check(
            "Create retrievers via factory",
            True,
            f"Created {len(retrievers)} retrievers",
        )

        return {
            "documents": documents,
            "vector_store": vector_store,
            "retrievers": retrievers,
        }

    except Exception as e:
        print_check("Correct initialization pattern", False, str(e))
        print(f"\n{Colors.RED}Traceback:{Colors.END}")
        print(traceback.format_exc())
        return None


# ==============================================================================
# 4. GRAPH COMPILATION VALIDATION
# ==============================================================================


def validate_graph_compilation(components: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    """Validate LangGraph compilation using actual src/ modules"""
    print_section("4. GRAPH COMPILATION VALIDATION")

    if not components:
        print_error("Skipping - correct initialization pattern failed")
        return {}

    results = {}

    try:
        # Use actual src/ modules
        from src.graph import build_all_graphs

        print_info("Building graphs using src.graph...")

        # Build all graphs using the factory function
        retrievers = components["retrievers"]
        graphs = build_all_graphs(retrievers)

        # Validate each graph was compiled
        for retriever_name in retrievers.keys():
            results[f"compile_{retriever_name}"] = True
            print_check(f"Compile {retriever_name} graph", True)

        components["graphs"] = graphs

        return results

    except Exception as e:
        print_check("Graph compilation", False, str(e))
        print(f"\n{Colors.RED}Traceback:{Colors.END}")
        print(traceback.format_exc())
        return results


# ==============================================================================
# 5. FUNCTIONAL TESTING
# ==============================================================================


def run_functional_tests(components: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    """Run functional tests with actual queries"""
    print_section("5. FUNCTIONAL TESTING")

    if not components or "graphs" not in components:
        print_error("Skipping - graph compilation failed")
        return {}

    results = {}
    test_question = "What is GDELT?"

    print_info(f"Test question: '{test_question}'")

    from src.graph import invoke_for_benchmark

    for retriever_name, graph in components["graphs"].items():
        try:
            # Execute graph via benchmark adapter
            result = invoke_for_benchmark(graph, test_question)

            # Validate result structure
            assert "question" in result, "Missing 'question' in result"
            assert "context" in result, "Missing 'context' in result"
            assert "response" in result, "Missing 'response' in result"
            assert len(result["context"]) > 0, "Empty context retrieved"
            assert len(result["response"]) > 0, "Empty response generated"

            results[f"test_{retriever_name}"] = True
            print_check(
                f"{retriever_name:<20} functional test",
                True,
                f"Retrieved {len(result['context'])} contexts, response length: {len(result['response'])}",
            )

        except Exception as e:
            results[f"test_{retriever_name}"] = False
            print_check(f"{retriever_name:<20} functional test", False, str(e))

    return results


# ==============================================================================
# 6. DIAGNOSTIC REPORT
# ==============================================================================


def generate_diagnostic_report(
    env_results: Dict[str, bool],
    import_results: Dict[str, Dict[str, Any]],
    pattern_results: Optional[Dict[str, Any]],
    graph_results: Dict[str, bool],
    functional_results: Dict[str, bool],
) -> bool:
    """Generate final diagnostic report"""
    print_section("6. DIAGNOSTIC REPORT")

    # Count passes/fails
    total_checks = 0
    passed_checks = 0

    for result_dict in [env_results, graph_results, functional_results]:
        for passed in result_dict.values():
            total_checks += 1
            if passed:
                passed_checks += 1

    # Import results counted separately
    for module_name, result in import_results.items():
        total_checks += 1
        if result["success"]:
            passed_checks += 1

    pass_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0

    print(f"{Colors.BOLD}Overall Results:{Colors.END}")
    print(f"  Total checks: {total_checks}")
    print(f"  Passed: {Colors.GREEN}{passed_checks}{Colors.END}")
    print(f"  Failed: {Colors.RED}{total_checks - passed_checks}{Colors.END}")
    print(
        f"  Pass rate: {Colors.GREEN if pass_rate == 100 else Colors.YELLOW}{pass_rate:.1f}%{Colors.END}"
    )

    # Identify critical issues
    print(f"\n{Colors.BOLD}Critical Issues:{Colors.END}")

    critical_issues = []

    # Check for broken imports
    failed_imports = [
        name for name, result in import_results.items() if not result["success"]
    ]
    if failed_imports:
        critical_issues.append(
            {
                "title": "Broken Module Imports",
                "severity": "HIGH",
                "modules": failed_imports,
                "impact": "src/ directory is not usable as a module",
                "fix": "Refactor src/retrievers.py to use factory pattern (see section 3 output)",
            }
        )

    # Check for missing environment
    if not env_results.get("openai_key"):
        critical_issues.append(
            {
                "title": "Missing OPENAI_API_KEY",
                "severity": "HIGH",
                "impact": "Cannot run LLM or generate embeddings",
                "fix": "Set OPENAI_API_KEY environment variable",
            }
        )

    if not env_results.get("qdrant"):
        critical_issues.append(
            {
                "title": "Qdrant Not Accessible",
                "severity": "HIGH",
                "impact": "Cannot store or retrieve document embeddings",
                "fix": "Start Qdrant: docker-compose up -d qdrant",
            }
        )

    if critical_issues:
        for i, issue in enumerate(critical_issues, 1):
            print(
                f"\n{Colors.RED}Issue {i}: {issue['title']} [SEVERITY: {issue['severity']}]{Colors.END}"
            )
            if "modules" in issue:
                print(f"  Affected modules: {', '.join(issue['modules'])}")
            print(f"  Impact: {issue['impact']}")
            print(f"  {Colors.GREEN}Fix: {issue['fix']}{Colors.END}")
    else:
        print(f"{Colors.GREEN}No critical issues found!{Colors.END}")

    # Recommendations
    print(f"\n{Colors.BOLD}Recommendations:{Colors.END}")

    if "src.retrievers" in failed_imports:
        print(f"""
{Colors.YELLOW}1. Fix src/retrievers.py:{Colors.END}

   Problem: Retrievers are created at module import time, but 'documents' doesn't exist yet.

   Solution: Convert to factory pattern

   {Colors.BLUE}# In src/retrievers.py - REPLACE module-level variables with factory function:{Colors.END}

   def create_retrievers(documents, vector_store):
       baseline_retriever = vector_store.as_retriever(search_kwargs={{"k": 5}})
       bm25_retriever = BM25Retriever.from_documents(documents, k=5)
       # ... etc
       return {{
           'baseline': baseline_retriever,
           'bm25': bm25_retriever,
           'ensemble': ensemble_retriever,
           'cohere_rerank': compression_retriever
       }}
""")

    if "src.graph" in failed_imports and "src.retrievers" in failed_imports:
        print(f"""
{Colors.YELLOW}2. Fix src/graph.py:{Colors.END}

   Problem: Depends on broken src/retrievers.py

   Solution: Update imports after fixing retrievers.py, or use factory pattern here too
""")

    if pattern_results and all([r for r in functional_results.values()]):
        print(f"""
{Colors.GREEN}3. Migration Path to src/ Module:{Colors.END}

   All validations passed! The correct initialization pattern is working.

   Next steps:
   1. Create src/factories.py with retriever factory functions
   2. Create src/loader.py with document loading utilities
   3. Update src/retrievers.py to export factory function (not instances)
   4. Update src/graph.py to use factories
   5. Refactor scripts/run_full_evaluation.py to import from src/

   See section 3 output above for working factory pattern example.
""")

    # Final status
    all_passed = pass_rate == 100.0
    return all_passed


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================


def main() -> int:
    """Main validation execution"""
    print(f"""
{Colors.BOLD}{Colors.BLUE}╔════════════════════════════════════════════════════════════════════════════════╗
║                    LANGGRAPH CONFIGURATION VALIDATION                          ║
║                                                                                ║
║  Purpose: Validate src/ module implementation and demonstrate correct patterns ║
╚════════════════════════════════════════════════════════════════════════════════╝{Colors.END}
""")

    # Run all validation stages
    env_results = check_environment()
    import_results = test_module_imports()
    pattern_components = demonstrate_correct_pattern()
    graph_results = validate_graph_compilation(pattern_components)
    functional_results = run_functional_tests(pattern_components)

    # Generate final report
    all_passed = generate_diagnostic_report(
        env_results,
        import_results,
        pattern_components,
        graph_results,
        functional_results,
    )

    # Print final status
    print_section("VALIDATION COMPLETE")
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ ALL VALIDATIONS PASSED{Colors.END}")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ VALIDATION FAILURES DETECTED{Colors.END}")
        print(
            f"\n{Colors.YELLOW}Review the diagnostic report above for fixes.{Colors.END}"
        )
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
