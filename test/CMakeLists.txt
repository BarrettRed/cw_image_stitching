add_executable(ssim_test ssim_test.cpp)
target_link_libraries(ssim_test PRIVATE ${OpenCV_LIBS})
install(TARGETS ssim_test DESTINATION bin)

add_executable(prepare_split_images prepare_split_images.cpp)
target_link_libraries(prepare_split_images PRIVATE ${OpenCV_LIBS})
install(TARGETS prepare_split_images DESTINATION bin)
