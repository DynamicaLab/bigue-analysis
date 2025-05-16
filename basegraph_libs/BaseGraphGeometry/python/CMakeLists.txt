find_package(pybind11 REQUIRED)

pybind11_add_module(_${EXTENSION_NAME} MODULE pybind.cpp)
# target_link_libraries(_${EXTENSION_NAME} PRIVATE BaseGraph::core ${EXTENSION_NAME})
target_include_directories(_${EXTENSION_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/../BaseGraph/include)

install(TARGETS _${EXTENSION_NAME} DESTINATION ./basegraph/${EXTENSION_NAME})
install(FILES ${EXTENSION_NAME}/__init__.py DESTINATION ./basegraph/${EXTENSION_NAME})
