module automated_trading_test_environment
"""
    greet(name::String)

    Friendly Hello world greeting function.

    Describe the function in between the triple quotes. 
    When adding new functions to your package, make sure to add the name of the function in the index.md as: 
    ```@docs
    automated_trading_test_environment.greet
    ```

"""
function greet(name::String)
    return "Hello, $name"*"!"
end

end