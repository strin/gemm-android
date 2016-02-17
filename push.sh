if [ -z $2 ]
then
  TEST_PATH=/sdcard/blas
else
  TEST_PATH=$2
fi

adb push intel-gemm/build/intelgemm $TEST_PATH
if [ -z $1 ]
then
  adb push gemm.cl $TEST_PATH/gemm.cl
else
  adb push $1 $TEST_PATH/gemm.cl
fi
