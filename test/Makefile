# Script arguments

SRC_DIRS := .
BUILD_DIR := .
INC_DIRS := ../src/cpp

PIP_CACHE_FILES := $(shell find .. -name '*.so' -o -wholename '*/__pycache__' -o -wholename '*.egg-info' -o -wholename '../build')

# Find all the C and C++ files we want to compile
# Note the single quotes around the * expressions. The shell will incorrectly expand these otherwise, but we want to send the * directly to the find command.
SRCS := $(shell find $(SRC_DIRS) -name '*.cpp')
# Add a prefix to INC_DIRS. So moduleA would become -ImoduleA. GCC understands this -I flag
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CPPFLAGS := $(INC_FLAGS) -std=c++17 -Wall # -MMD -MP

TESTERS := $(SRCS:.cpp=.out)

$(BUILD_DIR)/%.out: %.cpp
	@$(CXX) $< -o $@ $(INC_FLAGS) $(CPPFLAGS)

.PHONY: clean test
clean:
	@rm -rf $(BUILD_DIR)/*.out $(PIP_CACHE_FILES)
	
test: $(TESTERS)
	@$(foreach file, $^, $(BUILD_DIR)/$(file);)