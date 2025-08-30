cmake_minimum_required(VERSION 3.16)

project(Food-Journaling-App VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 REQUIRED COMPONENTS Quick)
find_library(ONNXRUNTIME_LIB onnxruntime HINTS /opt/homebrew/lib)


qt_standard_project_setup(REQUIRES 6.5)

qt_add_executable(appFood-Journaling-App
    main.cpp
)

qt_add_qml_module(appFood-Journaling-App
    URI Food-Journaling-App
    VERSION 1.0
    QML_FILES
        Main.qml
        UI-Items/PredictionCard.qml
    SOURCES
        backend/nutritionpredictor.cpp backend/nutritionpredictor.h
        backend/textpreprocessor.cpp backend/textpreprocessor.h
        controller/appcontroller.cpp controller/appcontroller.h
)

include_directories(/opt/homebrew/include/onnxruntime)

set(MODEL_FILES
    ${CMAKE_SOURCE_DIR}/models/nutrition_model.onnx
    ${CMAKE_SOURCE_DIR}/models/vectorizer_vocab.json
    ${CMAKE_SOURCE_DIR}/models/output_scaler_params.json
)

if(ANDROID)
    message(STATUS "Configuring for Android")
    # Point these to your cross-compiled ONNX Runtime build output for Android
    set(ONNX_ANDROID_ROOT ${CMAKE_SOURCE_DIR}/third_party/onnxruntime-android)
    link_directories(${ONNX_ANDROID_ROOT})

    # Use the shared lib on Android
    set(ONNXRUNTIME_LIB ${ONNX_ANDROID_ROOT}/libonnxruntime.so)

    # Use a dedicated android/ folder inside build tree for assets
    set(ANDROID_PACKAGE_SOURCE_DIR ${CMAKE_BINARY_DIR}/android)

    add_custom_command(TARGET appFood-Journaling-App POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${ONNX_ANDROID_ROOT}/libonnxruntime.so
        ${ANDROID_PACKAGE_SOURCE_DIR}/libs/arm64-v8a/libonnxruntime.so)

    # Ensure assets folder exists
    file(MAKE_DIRECTORY ${ANDROID_PACKAGE_SOURCE_DIR}/assets)

    # Copy model files into assets
    foreach(model_file ${MODEL_FILES})
        get_filename_component(model_name ${model_file} NAME)
        configure_file(${model_file} ${ANDROID_PACKAGE_SOURCE_DIR}/assets/${model_name} COPYONLY)
    endforeach()

    # Tell Qt to package this folder into the APK
    set_property(TARGET appFood-Journaling-App PROPERTY
        QT_ANDROID_PACKAGE_SOURCE_DIR ${ANDROID_PACKAGE_SOURCE_DIR})

endif()

target_link_libraries(appFood-Journaling-App
    PRIVATE Qt6::Quick
    ${ONNXRUNTIME_LIB}
)

include(GNUInstallDirs)
install(TARGETS appFood-Journaling-App
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
