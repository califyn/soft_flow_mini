#!/usr/bin/env python

import collections
import cupy
import cupy as cp
import os
import re
import torch
import typing


##########################################################


objCudacache = {}


def cuda_int32(intIn: int):
    return cupy.int32(intIn)


# end


def cuda_float32(fltIn: float):
    return cupy.float32(fltIn)


# end


def cuda_kernel(strFunction: str, strKernel: str, objVariables: typing.Dict):
    if "device" not in objCudacache:
        objCudacache["device"] = torch.cuda.get_device_name()
    # end

    strKey = strFunction

    for strVariable in objVariables:
        objValue = objVariables[strVariable]

        strKey += strVariable

        if objValue is None:
            continue

        elif type(objValue) == int:
            strKey += str(objValue)

        elif type(objValue) == float:
            strKey += str(objValue)

        elif type(objValue) == bool:
            strKey += str(objValue)

        elif type(objValue) == str:
            strKey += objValue

        elif type(objValue) == torch.Tensor:
            strKey += str(objValue.dtype)
            strKey += str(objValue.shape)
            strKey += str(objValue.stride())

        elif True:
            print(strVariable, type(objValue))
            assert False

        # end
    # end

    strKey += objCudacache["device"]

    if strKey not in objCudacache:
        for strVariable in objVariables:
            objValue = objVariables[strVariable]

            if objValue is None:
                continue

            elif type(objValue) == int:
                strKernel = strKernel.replace("{{" + strVariable + "}}", str(objValue))

            elif type(objValue) == float:
                strKernel = strKernel.replace("{{" + strVariable + "}}", str(objValue))

            elif type(objValue) == bool:
                strKernel = strKernel.replace("{{" + strVariable + "}}", str(objValue))

            elif type(objValue) == str:
                strKernel = strKernel.replace("{{" + strVariable + "}}", objValue)

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.uint8:
                strKernel = strKernel.replace("{{type}}", "unsigned char")

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.float16:
                strKernel = strKernel.replace("{{type}}", "half")

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.float32:
                strKernel = strKernel.replace("{{type}}", "float")

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.float64:
                strKernel = strKernel.replace("{{type}}", "double")

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.int32:
                strKernel = strKernel.replace("{{type}}", "int")

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.int64:
                strKernel = strKernel.replace("{{type}}", "long")

            elif type(objValue) == torch.Tensor:
                print(strVariable, objValue.dtype)
                assert False

            elif True:
                print(strVariable, type(objValue))
                assert False

            # end
        # end

        while True:
            objMatch = re.search("(SIZE_)([0-4])(\()([^\)]*)(\))", strKernel)

            if objMatch is None:
                break
            # end

            intArg = int(objMatch.group(2))

            strTensor = objMatch.group(4)
            intSizes = objVariables[strTensor].size()

            strKernel = strKernel.replace(
                objMatch.group(),
                str(
                    intSizes[intArg]
                    if torch.is_tensor(intSizes[intArg]) == False
                    else intSizes[intArg].item()
                ),
            )
        # end

        while True:
            objMatch = re.search("(OFFSET_)([0-4])(\()", strKernel)

            if objMatch is None:
                break
            # end

            intStart = objMatch.span()[1]
            intStop = objMatch.span()[1]
            intParentheses = 1

            while True:
                intParentheses += 1 if strKernel[intStop] == "(" else 0
                intParentheses -= 1 if strKernel[intStop] == ")" else 0

                if intParentheses == 0:
                    break
                # end

                intStop += 1
            # end

            intArgs = int(objMatch.group(2))
            strArgs = strKernel[intStart:intStop].split(",")

            assert intArgs == len(strArgs) - 1

            strTensor = strArgs[0]
            intStrides = objVariables[strTensor].stride()

            strIndex = []

            for intArg in range(intArgs):
                strIndex.append(
                    "(("
                    + strArgs[intArg + 1].replace("{", "(").replace("}", ")").strip()
                    + ")*"
                    + str(
                        intStrides[intArg]
                        if torch.is_tensor(intStrides[intArg]) == False
                        else intStrides[intArg].item()
                    )
                    + ")"
                )
            # end

            strKernel = strKernel.replace(
                "OFFSET_" + str(intArgs) + "(" + strKernel[intStart:intStop] + ")",
                "(" + str.join("+", strIndex) + ")",
            )
        # end

        while True:
            objMatch = re.search("(VALUE_)([0-4])(\()", strKernel)

            if objMatch is None:
                break
            # end

            intStart = objMatch.span()[1]
            intStop = objMatch.span()[1]
            intParentheses = 1

            while True:
                intParentheses += 1 if strKernel[intStop] == "(" else 0
                intParentheses -= 1 if strKernel[intStop] == ")" else 0

                if intParentheses == 0:
                    break
                # end

                intStop += 1
            # end

            intArgs = int(objMatch.group(2))
            strArgs = strKernel[intStart:intStop].split(",")

            assert intArgs == len(strArgs) - 1

            strTensor = strArgs[0]
            intStrides = objVariables[strTensor].stride()

            strIndex = []

            for intArg in range(intArgs):
                strIndex.append(
                    "(("
                    + strArgs[intArg + 1].replace("{", "(").replace("}", ")").strip()
                    + ")*"
                    + str(
                        intStrides[intArg]
                        if torch.is_tensor(intStrides[intArg]) == False
                        else intStrides[intArg].item()
                    )
                    + ")"
                )
            # end

            strKernel = strKernel.replace(
                "VALUE_" + str(intArgs) + "(" + strKernel[intStart:intStop] + ")",
                strTensor + "[" + str.join("+", strIndex) + "]",
            )
        # end

        objCudacache[strKey] = {"strFunction": strFunction, "strKernel": strKernel}
    # end

    return strKey


# end


@cupy.memoize(for_each_device=True)
def cuda_launch(strKey: str):
    if "CUDA_HOME" not in os.environ:
        os.environ["CUDA_HOME"] = cupy.cuda.get_cuda_path()
    # end

    """
    return cupy.cuda.compile_with_cache(
        objCudacache[strKey]["strKernel"],
        tuple(
            [
                "-I " + os.environ["CUDA_HOME"],
                "-I " + os.environ["CUDA_HOME"] + "/include",
            ]
        ),
    ).get_function(objCudacache[strKey]["strFunction"])
    """

    add_kernel = cp.RawKernel(objCudacache[strKey]["strKernel"], objCudacache[strKey]["strFunction"])

    return add_kernel

# end


##########################################################


def softsplat(
        tenIn: torch.Tensor, tenFlow: torch.Tensor, tenMetric: torch.Tensor, strMode: str
):
    return _softsplat_downsample(tenIn, tenFlow, tenMetric, strMode, scale=1)

def downsample(
        tenIn: torch.Tensor, tenMetric: torch.Tensor, strMode: str, scale=1
):
    return _softsplat_downsample(tenIn, torch.zeros((tenIn.shape[0], 2, tenIn.shape[2], tenIn.shape[3])).to(tenIn.device), tenMetric, strMode, scale)

def _softsplat_downsample(
        tenIn: torch.Tensor, tenFlow: torch.Tensor, tenMetric: torch.Tensor, strMode: str, scale: int
):
    assert strMode.split("-")[0] in ["sum", "linear", "soft", "linear_unn"]

    if strMode == "sum":
        assert tenMetric is None
    if strMode == "avg":
        assert tenMetric is None
    if strMode.split("-")[0] == "linear" or strMode.split("-")[0] == "linear_unn":
        assert tenMetric is not None
    if strMode.split("-")[0] == "soft":
        assert tenMetric is not None

    if strMode == "avg" or strMode == "sum":
        tenIn = torch.cat(
            [
                tenIn,
                tenIn.new_ones([tenIn.shape[0], 1, tenIn.shape[2], tenIn.shape[3]]),
            ],
            1,
        )
    elif strMode.split("-")[0] == "linear" or strMode.split("-")[0] == "linear_unn":
        tenIn = torch.cat([tenIn * tenMetric, tenMetric], 1)
    elif strMode.split("-")[0] == "soft":
        tenIn = torch.cat([tenIn * tenMetric.exp(), tenMetric.exp()], 1)

    return softsplat_func.apply(tenIn, tenFlow, scale)

# end


class softsplat_func(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, tenIn, tenFlow, scale):
        tenOut = tenIn.new_zeros(
            [tenIn.shape[0], tenIn.shape[1], (tenIn.shape[2] // scale) * scale, (tenIn.shape[3] // scale) * scale]
        )

        if tenIn.is_cuda == True:
            cuda_launch(
                cuda_kernel(
                    "softsplat_out",
                    """
                extern "C" __global__ void __launch_bounds__(512) softsplat_out(
                    const int n,
                    const {{type}}* __restrict__ tenIn,
                    const {{type}}* __restrict__ tenFlow,
                    {{type}}* __restrict__ tenOut,
                    const int scale
                ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
                    const int intN = ( intIndex / SIZE_3(tenIn) / SIZE_2(tenIn) / SIZE_1(tenIn) ) % SIZE_0(tenIn);
                    const int intC = ( intIndex / SIZE_3(tenIn) / SIZE_2(tenIn)                  ) % SIZE_1(tenIn);
                    const int intY = ( intIndex / SIZE_3(tenIn)                                   ) % SIZE_2(tenIn);
                    const int intX = ( intIndex                                                    ) % SIZE_3(tenIn);

                    {{type}} fltIn = VALUE_4(tenIn, intN, intC, intY, intX);

                    assert(SIZE_1(tenFlow) == 2);

                    for (int offset_x = 0; offset_x < scale; offset_x += 1) {
                        for (int offset_y = 0; offset_y < scale; offset_y += 1) {
                            {{type}} fltX = ({{type}}) (intX) + VALUE_4(tenFlow, intN, 0, intY, intX);
                            {{type}} fltY = ({{type}}) (intY) + VALUE_4(tenFlow, intN, 1, intY, intX);

                            if (isfinite(fltX) == false) { return; }
                            if (isfinite(fltY) == false) { return; }

                            if (fltX >= ({{type}}) (SIZE_3(tenIn)) - 1.0 && scale > 1) {
                               fltX = fltX + (fltX - ({{type}}) (SIZE_3(tenIn)) + 1.0) * ({{type}}) ((abs(offset_x - (SIZE_3(tenIn) % scale))) % scale);
                               fltX = (fltX - offset_x) / scale;
                            } else if (fltX - offset_x < 0.0) {
                                fltX = fltX - offset_x; 
                            } else {
                                fltX = (fltX - offset_x) / scale;
                            }

                            if (fltY >= ({{type}}) (SIZE_2(tenIn)) - 1.0 && scale > 1) {
                               fltY = fltY + (fltY - ({{type}}) (SIZE_2(tenIn)) + 1.0) * ({{type}}) ((abs(offset_y - (SIZE_2(tenIn) % scale))) % scale);
                               fltY = (fltY - offset_y) / scale;
                            } else if (fltY - offset_y < 0.0) {
                                fltY = fltY - offset_y; 
                            } else {
                                fltY = (fltY - offset_y) / scale;
                            }

                            int truIntNorthwestX = (int) (floor(fltX));
                            int truIntNorthwestY = (int) (floor(fltY));
                            int truIntNortheastX = truIntNorthwestX + 1;
                            int truIntNortheastY = truIntNorthwestY;
                            int truIntSouthwestX = truIntNorthwestX;
                            int truIntSouthwestY = truIntNorthwestY + 1;
                            int truIntSoutheastX = truIntNorthwestX + 1;
                            int truIntSoutheastY = truIntNorthwestY + 1;

                            int intNorthwestX = (int) (floor(fltX)) * scale + offset_x;
                            int intNorthwestY = (int) (floor(fltY)) * scale + offset_y;
                            int intNortheastX = intNorthwestX + scale;
                            int intNortheastY = intNorthwestY;
                            int intSouthwestX = intNorthwestX;
                            int intSouthwestY = intNorthwestY + scale;
                            int intSoutheastX = intNorthwestX + scale;
                            int intSoutheastY = intNorthwestY + scale;

                            {{type}} fltNorthwest = (({{type}}) (truIntSoutheastX) - fltX) * (({{type}}) (truIntSoutheastY) - fltY);
                            {{type}} fltNortheast = (fltX - ({{type}}) (truIntSouthwestX)) * (({{type}}) (truIntSouthwestY) - fltY);
                            {{type}} fltSouthwest = (({{type}}) (truIntNortheastX) - fltX) * (fltY - ({{type}}) (truIntNortheastY));
                            {{type}} fltSoutheast = (fltX - ({{type}}) (truIntNorthwestX)) * (fltY - ({{type}}) (truIntNorthwestY));

                            if ((truIntNorthwestX >= 0) && (truIntNorthwestX < SIZE_3(tenOut) / scale) && (truIntNorthwestY >= 0) && (truIntNorthwestY < SIZE_2(tenOut) / scale)) {
                                atomicAdd(&tenOut[OFFSET_4(tenOut, intN, intC, intNorthwestY, intNorthwestX)], fltIn * fltNorthwest);
                            }

                            if ((truIntNortheastX >= 0) && (truIntNortheastX < SIZE_3(tenOut) / scale) && (truIntNortheastY >= 0) && (truIntNortheastY < SIZE_2(tenOut) / scale)) {
                                atomicAdd(&tenOut[OFFSET_4(tenOut, intN, intC, intNortheastY, intNortheastX)], fltIn * fltNortheast);
                            }

                            if ((truIntSouthwestX >= 0) && (truIntSouthwestX < SIZE_3(tenOut) / scale) && (truIntSouthwestY >= 0) && (truIntSouthwestY < SIZE_2(tenOut) / scale)) {
                                atomicAdd(&tenOut[OFFSET_4(tenOut, intN, intC, intSouthwestY, intSouthwestX)], fltIn * fltSouthwest);
                            }

                            if ((truIntSoutheastX >= 0) && (truIntSoutheastX < SIZE_3(tenOut) / scale) && (truIntSoutheastY >= 0) && (truIntSoutheastY < SIZE_2(tenOut) / scale)) {
                                atomicAdd(&tenOut[OFFSET_4(tenOut, intN, intC, intSoutheastY, intSoutheastX)], fltIn * fltSoutheast);
                            }
                        }
                    }
                } }
            """,
                    {"tenIn": tenIn, "tenFlow": tenFlow, "tenOut": tenOut},
                )
            )(
                grid=tuple([int((tenIn.nelement() + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    cuda_int32(tenIn.nelement()),
                    tenIn.data_ptr(),
                    tenFlow.data_ptr(),
                    tenOut.data_ptr(),
                    scale
                ],
                stream=collections.namedtuple("Stream", "ptr")(
                    torch.cuda.current_stream().cuda_stream
                ),
            )

        elif tenIn.is_cuda != True:
            assert False

        # end

        self.save_for_backward(tenIn, tenFlow)
        self.scale = scale

        return tenOut

    # end

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(self, tenOutgrad):
        tenIn, tenFlow = self.saved_tensors
        scale = self.scale

        tenOutgrad = tenOutgrad.contiguous()
        assert tenOutgrad.is_cuda == True

        tenIngrad = (
            tenIn.new_zeros(
                [tenIn.shape[0], tenIn.shape[1], tenIn.shape[2], tenIn.shape[3]]
            )
            if self.needs_input_grad[0] == True
            else None
        )
        tenFlowgrad = (
            tenFlow.new_zeros(
                [tenFlow.shape[0], tenFlow.shape[1], tenFlow.shape[2], tenFlow.shape[3]]
            )
            if self.needs_input_grad[1] == True
            else None
        )

        if tenIngrad is not None:
            cuda_launch(
                cuda_kernel(
                    "softsplat_ingrad",
                    """
                extern "C" __global__ void __launch_bounds__(512) softsplat_ingrad(
                    const int n,
                    const {{type}}* __restrict__ tenIn,
                    const {{type}}* __restrict__ tenFlow,
                    const {{type}}* __restrict__ tenOutgrad,
                    {{type}}* __restrict__ tenIngrad,
                    {{type}}* __restrict__ tenFlowgrad,
                    const int scale
                ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
                    const int intN = ( intIndex / SIZE_3(tenIngrad) / SIZE_2(tenIngrad) / SIZE_1(tenIngrad) ) % SIZE_0(tenIngrad);
                    const int intC = ( intIndex / SIZE_3(tenIngrad) / SIZE_2(tenIngrad)                     ) % SIZE_1(tenIngrad);
                    const int intY = ( intIndex / SIZE_3(tenIngrad)                                         ) % SIZE_2(tenIngrad);
                    const int intX = ( intIndex                                                             ) % SIZE_3(tenIngrad);

                    assert(SIZE_1(tenFlow) == 2);

                    {{type}} fltIngrad = 0.0f;

                    for (int offset_x = 0; offset_x < scale; offset_x += 1) {
                        for (int offset_y = 0; offset_y < scale; offset_y += 1) {
                            {{type}} fltX = ({{type}}) (intX) + VALUE_4(tenFlow, intN, 0, intY, intX);
                            {{type}} fltY = ({{type}}) (intY) + VALUE_4(tenFlow, intN, 1, intY, intX);

                            if (isfinite(fltX) == false) { return; }
                            if (isfinite(fltY) == false) { return; }

                            if (fltX >= ({{type}}) (SIZE_3(tenIn)) - 1.0) {
                               fltX = fltX + (fltX - ({{type}}) (SIZE_3(tenIn)) + 1.0) * ({{type}}) ((abs(offset_x - (SIZE_3(tenIn) % scale))) % scale);
                               fltX = fltX + (fltX - ({{type}}) (SIZE_3(tenIn)) + 1.0) * ({{type}}) (offset_x);
                               fltX = (fltX - offset_x) / scale;
                            } else if (fltX - offset_x < 0.0) {
                                fltX = fltX - offset_x; 
                            } else {
                                fltX = (fltX - offset_x) / scale;
                            }

                            if (fltY >= ({{type}}) (SIZE_2(tenIn)) - 1.0) {
                               fltY = fltY + (fltY - ({{type}}) (SIZE_2(tenIn)) + 1.0) * ({{type}}) ((abs(offset_y - (SIZE_2(tenIn) % scale))) % scale);
                               fltY = (fltY - offset_y) / scale;
                            } else if (fltY - offset_y < 0.0) {
                                fltY = fltY - offset_y; 
                            } else {
                                fltY = (fltY - offset_y) / scale;
                            }

                            int truIntNorthwestX = (int) (floor(fltX));
                            int truIntNorthwestY = (int) (floor(fltY));
                            int truIntNortheastX = truIntNorthwestX + 1;
                            int truIntNortheastY = truIntNorthwestY;
                            int truIntSouthwestX = truIntNorthwestX;
                            int truIntSouthwestY = truIntNorthwestY + 1;
                            int truIntSoutheastX = truIntNorthwestX + 1;
                            int truIntSoutheastY = truIntNorthwestY + 1;

                            int intNorthwestX = (int) (floor(fltX)) * scale + offset_x;
                            int intNorthwestY = (int) (floor(fltY)) * scale + offset_y;
                            int intNortheastX = intNorthwestX + scale;
                            int intNortheastY = intNorthwestY;
                            int intSouthwestX = intNorthwestX;
                            int intSouthwestY = intNorthwestY + scale;
                            int intSoutheastX = intNorthwestX + scale;
                            int intSoutheastY = intNorthwestY + scale;

                            {{type}} fltNorthwest = (({{type}}) (truIntSoutheastX) - fltX) * (({{type}}) (truIntSoutheastY) - fltY);
                            {{type}} fltNortheast = (fltX - ({{type}}) (truIntSouthwestX)) * (({{type}}) (truIntSouthwestY) - fltY);
                            {{type}} fltSouthwest = (({{type}}) (truIntNortheastX) - fltX) * (fltY - ({{type}}) (truIntNortheastY));
                            {{type}} fltSoutheast = (fltX - ({{type}}) (truIntNorthwestX)) * (fltY - ({{type}}) (truIntNorthwestY));

                            if ((truIntNorthwestX >= 0) && (truIntNorthwestX < SIZE_3(tenOutgrad) / scale) && (truIntNorthwestY >= 0) && (truIntNorthwestY < SIZE_2(tenOutgrad) / scale)) {
                                fltIngrad += VALUE_4(tenOutgrad, intN, intC, intNorthwestY, intNorthwestX) * fltNorthwest;
                            }

                            if ((truIntNortheastX >= 0) && (truIntNortheastX < SIZE_3(tenOutgrad) / scale) && (truIntNortheastY >= 0) && (truIntNortheastY < SIZE_2(tenOutgrad) / scale)) {
                                fltIngrad += VALUE_4(tenOutgrad, intN, intC, intNortheastY, intNortheastX) * fltNortheast;
                            }

                            if ((truIntSouthwestX >= 0) && (truIntSouthwestX < SIZE_3(tenOutgrad) / scale) && (truIntSouthwestY >= 0) && (truIntSouthwestY < SIZE_2(tenOutgrad) / scale)) {
                                fltIngrad += VALUE_4(tenOutgrad, intN, intC, intSouthwestY, intSouthwestX) * fltSouthwest;
                            }

                            if ((truIntSoutheastX >= 0) && (truIntSoutheastX < SIZE_3(tenOutgrad) / scale) && (truIntSoutheastY >= 0) && (truIntSoutheastY < SIZE_2(tenOutgrad) / scale)) {
                                fltIngrad += VALUE_4(tenOutgrad, intN, intC, intSoutheastY, intSoutheastX) * fltSoutheast;
                            }

                            tenIngrad[intIndex] = fltIngrad;
                        }
                    }
                } }
            """,
                    {
                        "tenIn": tenIn,
                        "tenFlow": tenFlow,
                        "tenOutgrad": tenOutgrad,
                        "tenIngrad": tenIngrad,
                        "tenFlowgrad": tenFlowgrad,
                    },
                )
            )(
                grid=tuple([int((tenIngrad.nelement() + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    cuda_int32(tenIngrad.nelement()),
                    tenIn.data_ptr(),
                    tenFlow.data_ptr(),
                    tenOutgrad.data_ptr(),
                    tenIngrad.data_ptr(),
                    None,
                    scale
                ],
                stream=collections.namedtuple("Stream", "ptr")(
                    torch.cuda.current_stream().cuda_stream
                ),
            )
        # end

        if tenFlowgrad is not None:
            cuda_launch(
                cuda_kernel(
                    "softsplat_flowgrad",
                    """
                extern "C" __global__ void __launch_bounds__(512) softsplat_flowgrad(
                    const int n,
                    const {{type}}* __restrict__ tenIn,
                    const {{type}}* __restrict__ tenFlow,
                    const {{type}}* __restrict__ tenOutgrad,
                    {{type}}* __restrict__ tenIngrad,
                    {{type}}* __restrict__ tenFlowgrad,
                    const int scale
                ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
                    const int intN = ( intIndex / SIZE_3(tenFlowgrad) / SIZE_2(tenFlowgrad) / SIZE_1(tenFlowgrad) ) % SIZE_0(tenFlowgrad);
                    const int intC = ( intIndex / SIZE_3(tenFlowgrad) / SIZE_2(tenFlowgrad)                       ) % SIZE_1(tenFlowgrad);
                    const int intY = ( intIndex / SIZE_3(tenFlowgrad)                                             ) % SIZE_2(tenFlowgrad);
                    const int intX = ( intIndex                                                                   ) % SIZE_3(tenFlowgrad);

                    assert(SIZE_1(tenFlow) == 2);

                    {{type}} fltFlowgrad = 0.0f;

                    for (int offset_x = 0; offset_x < scale; offset_x += 1) {
                        for (int offset_y = 0; offset_y < scale; offset_y += 1) {
                            {{type}} fltX = ({{type}}) (intX) + VALUE_4(tenFlow, intN, 0, intY, intX);
                            {{type}} fltY = ({{type}}) (intY) + VALUE_4(tenFlow, intN, 1, intY, intX);

                            if (isfinite(fltX) == false) { return; }
                            if (isfinite(fltY) == false) { return; }

                            {{type}} dfltXX = 0.0f; // freeze gradient, this isn't actually correct but stabilizes perf
                            {{type}} dfltYY = 0.0f;

                            if (fltX >= ({{type}}) (SIZE_3(tenIn)) - 1.0) {
                               fltX = fltX + (fltX - ({{type}}) (SIZE_3(tenIn)) + 1.0) * ({{type}}) ((abs(offset_x - (SIZE_3(tenIn) % scale))) % scale);
                               fltX = (fltX - offset_x) / scale;
                            } else if (fltX - offset_x < 0.0) {
                                fltX = fltX - offset_x; 
                            } else {
                                fltX = (fltX - offset_x) / scale;
                                dfltXX = 1.0f / scale;
                            }

                            if (fltY >= ({{type}}) (SIZE_2(tenIn)) - 1.0) {
                               fltY = fltY + (fltY - ({{type}}) (SIZE_2(tenIn)) + 1.0) * ({{type}}) (offset_y);
                               fltY = (fltY - offset_y) / scale;
                            } else if (fltY - offset_y < 0.0) {
                                fltY = fltY - offset_y; 
                            } else {
                                fltY = (fltY - offset_y) / scale;
                                dfltYY = 1.0f / scale;
                            }

                            int truIntNorthwestX = (int) (floor(fltX));
                            int truIntNorthwestY = (int) (floor(fltY));
                            int truIntNortheastX = truIntNorthwestX + 1;
                            int truIntNortheastY = truIntNorthwestY;
                            int truIntSouthwestX = truIntNorthwestX;
                            int truIntSouthwestY = truIntNorthwestY + 1;
                            int truIntSoutheastX = truIntNorthwestX + 1;
                            int truIntSoutheastY = truIntNorthwestY + 1;

                            int intNorthwestX = (int) (floor(fltX)) * scale + offset_x;
                            int intNorthwestY = (int) (floor(fltY)) * scale + offset_y;
                            int intNortheastX = intNorthwestX + scale;
                            int intNortheastY = intNorthwestY;
                            int intSouthwestX = intNorthwestX;
                            int intSouthwestY = intNorthwestY + scale;
                            int intSoutheastX = intNorthwestX + scale;
                            int intSoutheastY = intNorthwestY + scale;

                            {{type}} fltNorthwest = 0.0f;
                            {{type}} fltNortheast = 0.0f;
                            {{type}} fltSouthwest = 0.0f;
                            {{type}} fltSoutheast = 0.0f;

                            {{type}} dfltXYXY = 0.0f;
                            if (intC == 0) {
                                dfltXYXY = dfltYY;
                                fltNorthwest = (({{type}}) (-1.0f)) * (({{type}}) (truIntSoutheastY) - fltY);
                                fltNortheast = (({{type}}) (+1.0f)) * (({{type}}) (truIntSouthwestY) - fltY);
                                fltSouthwest = (({{type}}) (-1.0f)) * (fltY - ({{type}}) (truIntNortheastY));
                                fltSoutheast = (({{type}}) (+1.0f)) * (fltY - ({{type}}) (truIntNorthwestY));

                            } else if (intC == 1) {
                                dfltXYXY = dfltXX;
                                fltNorthwest = (({{type}}) (truIntSoutheastX) - fltX) * (({{type}}) (-1.0f));
                                fltNortheast = (fltX - ({{type}}) (truIntSouthwestX)) * (({{type}}) (-1.0f));
                                fltSouthwest = (({{type}}) (truIntNortheastX) - fltX) * (({{type}}) (+1.0f));
                                fltSoutheast = (fltX - ({{type}}) (truIntNorthwestX)) * (({{type}}) (+1.0f));
                            }

                            for (int intChannel = 0; intChannel < SIZE_1(tenOutgrad); intChannel += 1) {
                                {{type}} fltIn = VALUE_4(tenIn, intN, intChannel, intY, intX);

                                if ((truIntNorthwestX >= 0) && (truIntNorthwestX < SIZE_3(tenOutgrad) / scale) && (truIntNorthwestY >= 0) && (truIntNorthwestY < SIZE_2(tenOutgrad) / scale)) {
                                    fltFlowgrad += VALUE_4(tenOutgrad, intN, intChannel, intNorthwestY, intNorthwestX) * fltIn * fltNorthwest * dfltXYXY;
                                }

                                if ((truIntNortheastX >= 0) && (truIntNortheastX < SIZE_3(tenOutgrad) / scale) && (truIntNortheastY >= 0) && (truIntNortheastY < SIZE_2(tenOutgrad) / scale)) {
                                    fltFlowgrad += VALUE_4(tenOutgrad, intN, intChannel, intNortheastY, intNortheastX) * fltIn * fltNortheast * dfltXYXY;
                                }

                                if ((truIntSouthwestX >= 0) && (truIntSouthwestX < SIZE_3(tenOutgrad) / scale) && (truIntSouthwestY >= 0) && (truIntSouthwestY < SIZE_2(tenOutgrad) / scale)) {
                                    fltFlowgrad += VALUE_4(tenOutgrad, intN, intChannel, intSouthwestY, intSouthwestX) * fltIn * fltSouthwest * dfltXYXY;
                                }

                                if ((truIntSoutheastX >= 0) && (truIntSoutheastX < SIZE_3(tenOutgrad) / scale) && (truIntSoutheastY >= 0) && (truIntSoutheastY < SIZE_2(tenOutgrad) / scale)) {
                                    fltFlowgrad += VALUE_4(tenOutgrad, intN, intChannel, intSoutheastY, intSoutheastX) * fltIn * fltSoutheast * dfltXYXY;
                                }
                            }

                            tenFlowgrad[intIndex] = fltFlowgrad;
                        }
                    }
                } }
            """,
                    {
                        "tenIn": tenIn,
                        "tenFlow": tenFlow,
                        "tenOutgrad": tenOutgrad,
                        "tenIngrad": tenIngrad,
                        "tenFlowgrad": tenFlowgrad,
                    },
                )
            )(
                grid=tuple([int((tenFlowgrad.nelement() + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    cuda_int32(tenFlowgrad.nelement()),
                    tenIn.data_ptr(),
                    tenFlow.data_ptr(),
                    tenOutgrad.data_ptr(),
                    None,
                    tenFlowgrad.data_ptr(),
                    scale
                ],
                stream=collections.namedtuple("Stream", "ptr")(
                    torch.cuda.current_stream().cuda_stream
                ),
            )
        # end

        return tenIngrad, tenFlowgrad, None

    # end


# end
