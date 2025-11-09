using Test
using automated_trading_test_enviroment  # Import your package

#add sets of tests under the @testset macro and individual tests under the @test macro
#When adding new unit tests and new external packages to your package updates to the ci.yml file may be needed
@testset "automated_trading_test_enviroment.jl Tests" begin
    @test automated_trading_test_enviroment.greet("Alice") == "Hello, Alice!"  # Example unit test
end