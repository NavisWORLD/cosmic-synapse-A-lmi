@echo off
echo ========================================
echo A-LMI System - Run All Tests
echo ========================================
echo.

echo Starting all integration tests...
echo.

cd a_lmi_system

echo [1/3] Running End-to-End Integration Tests...
pytest tests/test_integration.py -v

echo.
echo [2/3] Running VLCL Simulation Tests...
pytest tests/test_vlcl_simulation.py -v

echo.
echo [3/3] Running Autonomous Learning Tests...
pytest tests/test_autonomous_learning.py -v

echo.
echo ========================================
echo All tests completed!
echo ========================================
pause

