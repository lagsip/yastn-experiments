
def run_all_tests():
    from . import generator_test_snippet
    generator_test_snippet.run_tests()

    from . import test_generate_lindblad
    test_generate_lindblad.run_tests()

    from . import test_time_evolve
    test_time_evolve.run_tests()