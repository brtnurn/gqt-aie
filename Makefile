##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
# VITIS related variables
AIETOOLS_DIR ?= $(shell realpath $(dir $(shell which xchesscc))/../)
AIE_INCLUDE_DIR ?= ${AIETOOLS_DIR}/data/versal_prod/lib
AIE2_INCLUDE_DIR ?= ${AIETOOLS_DIR}/data/aie_ml/lib

AIEOPT_DIR ?= $(shell realpath $(dir $(shell which aie-opt))/..)

WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body

CHESSCC1_FLAGS = -f -p me -P ${AIE_INCLUDE_DIR} -I ${AIETOOLS_DIR}/include
CHESSCC2_FLAGS = -f -p me -P ${AIE2_INCLUDE_DIR} -I ${AIETOOLS_DIR}/include
CHESS_FLAGS = -P ${AIE_INCLUDE_DIR}

CHESSCCWRAP1_FLAGS = aie -I ${AIETOOLS_DIR}/include 
CHESSCCWRAP2_FLAGS = aie2 -I ${AIETOOLS_DIR}/include
CHESSCCWRAP2P_FLAGS = aie2p -I ${AIETOOLS_DIR}/include 
PEANOWRAP2_FLAGS = -O1 -std=c++20 --target=aie2-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include 
PEANOWRAP2P_FLAGS = -O1 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include 

TEST_POWERSHELL := $(shell command -v powershell.exe >/dev/null 2>&1 && echo yes || echo no)
ifeq ($(TEST_POWERSHELL),yes)
	powershell = powershell.exe
	getwslpath = wslpath -w
else
	powershell = 
	getwslpath = echo
endif


devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
targetname = add_wahbm

all: build/final.xclbin ${targetname}.exe

aie_py_src=${targetname}.py
use_placed?=0

ifeq (${use_placed}, 1)
aie_py_src=${targetname}_placed.py
endif

build/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${devicename} > $@

build/add_wahbm.cc.o: ${srcdir}/add_wahbm.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -c $< -o ${@F}
else ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c $< -o ${@F}
else
	echo "Device type not supported"
endif

build/final.xclbin: build/aie.mlir build/add_wahbm.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts.bin ${<:%=../%}
		
${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake `${getwslpath} ${srcdir}` -DTARGET_NAME=${targetname}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif


COUNT="1024"
run: ${targetname}.exe build/final.xclbin
	${powershell} ./$< -x build/final.xclbin -i build/insts.bin -k MLIR_AIE -v 1

clean:
	rm -rf build _build ${targetname}.exe