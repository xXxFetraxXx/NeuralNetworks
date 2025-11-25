# NeuralNetworksBeta - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2025 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import pytest
try:
    from NeuralNetworks.src.NeuralNetworks import *
    try:
        img, inputs, outputs = image_from_url("https://unesco.org.uk/site/assets/files/6266/the_forth_bridge_2.jpeg",1)
        for fourier in [False, True]:
            for compilation in [False, True]:
                try:
                    net = MLP([2,1,3],Fourier=fourier,optim="Adam", crit="MSE", norm="Relu",Iscompiled=compilation)
                    try:
                        net.train(inputs,outputs,1,1024)
                        try:
                            net
                        except Exception:
                            pytest.fail(f"MLP(Fourier={fourier},optim={optim}, crit={crit}, norm={norm},Iscompiled={compilation}) failed __repr__")
                            print(Exception)
                        try:
                            net(inputs)
                        except Exception:
                            pytest.fail(f"MLP(Fourier={fourier},optim={optim}, crit={crit}, norm={norm},Iscompiled={compilation}) failed __call__")
                            print(Exception)
                        try:
                            net.params()
                        except Exception:
                            pytest.fail(f"MLP(Fourier={fourier},optim={optim}, crit={crit}, norm={norm},Iscompiled={compilation}) failed params()")
                            print(Exception)
                        try:
                            net.neurons()
                        except Exception:
                            pytest.fail(f"MLP(Fourier={fourier},optim={optim}, crit={crit}, norm={norm},Iscompiled={compilation}) failed neurons()")
                            print(Exception)
                        try:
                            net.nb_params()
                        except Exception:
                            pytest.fail(f"MLP(Fourier={fourier},optim={optim}, crit={crit}, norm={norm},Iscompiled={compilation}) failed nb_params()")
                            print(Exception)
                        try:
                            net.plot(img,inputs)
                        except Exception:
                            pytest.fail(f"MLP(Fourier={fourier},optim={optim}, crit={crit}, norm={norm},Iscompiled={compilation}) failed plot()")
                            print(Exception)
                        net2 = MLP([2,1,3],Fourier=fourier,optim=optim, crit=crit, norm=norm,Iscompiled=compilation)
                        try:
                            train(inputs,outputs,1,1024,net,net2)
                            try:
                                plot(img,inputs,net,net2)
                            except Exception:
                                pytest.fail(f"MLP(Fourier={fourier},optim={optim}, crit={crit}, norm={norm},Iscompiled={compilation}) failed plot(*nets)")
                                print(Exception)
                            try:
                                compare(img,inputs,net,net2)
                            except Exception:
                                pytest.fail(f"MLP(Fourier={fourier},optim={optim}, crit={crit}, norm={norm},Iscompiled={compilation}) failed compare(*nets)")
                                print(Exception)
                            try:
                                losses(net,net2)
                            except Exception:
                                pytest.fail(f"MLP(Fourier={fourier},optim={optim}, crit={crit}, norm={norm},Iscompiled={compilation}) failed losses(*nets)")
                                print(Exception)
                        except Exception:
                            pytest.fail(f"MLP(Fourier={fourier},optim={optim}, crit={crit}, norm={norm},Iscompiled={compilation}) failed train(*nets)")
                            print(Exception)
                    except Exception:
                        pytest.fail(f"MLP(Fourier={fourier},optim={optim}, crit={crit}, norm={norm},Iscompiled={compilation}) failed train()")
                        print(Exception)
                except Exception:
                    pytest.fail(f"MLP(Fourier={fourier},optim={optim}, crit={crit}, norm={norm},Iscompiled={compilation}) failed __init__")
                    print(Exception)
        try:
            net3 = MLP(Fourier=False,optim="Random", crit="MSE", norm="Relu",Iscompiled=False)
        except Exception:
            pytest.fail(f"MLP(optim='Random') failed __init__")
            print(Exception)
        try:
            net3 = MLP(Fourier=False,optim="Adam", crit="Random", norm="Relu",Iscompiled=False)
        except Exception:
            pytest.fail(f"MLP(crit='Random') failed __init__")
            print(Exception)
        try:
            net3 = MLP(Fourier=False,optim="Adam", crit="MSE", norm="Random",Iscompiled=False)
        except Exception:
            pytest.fail(f"MLP(norm='Random') failed __init__")
            print(Exception)
        try:
            net.train(inputs,outputs,1,64)
        except Exception:
            pytest.fail(f"MLP failed train() second pass")
            print(Exception)
        try:
            net
        except Exception:
            pytest.fail(f"MLP failed __repr__ second pass")
            print(Exception)
        try:
            net(inputs)
        except Exception:
            pytest.fail(f"MLP failed __call__ second pass")
            print(Exception)
        try:
            net.params()
        except Exception:
            pytest.fail(f"MLP failed params() second pass")
            print(Exception)
        try:
            net.neurons()
        except Exception:
            pytest.fail(f"MLP failed neurons() second pass")
            print(Exception)
        try:
            net.nb_params()
        except Exception:
            pytest.fail(f"MLP failed nb_params() second pass")
            print(Exception)
        try:
            net.plot(img,inputs)
        except Exception:
            pytest.fail(f"MLP failed plot() second pass")
            print(Exception)
        try:
            train(inputs,outputs,1,64,net,net2)
        except Exception:
            pytest.fail(f"MLP failed train(*nets) second pass")
            print(Exception)
        try:
            plot(img,inputs,net,net2)
        except Exception:
            pytest.fail(f"MLP failed plot(*nets) second pass")
            print(Exception)
        try:
            compare(img,inputs,net,net2)
        except Exception:
            pytest.fail(f"MLP failed compare(*nets) second pass")
            print(Exception)
        try:
            losses(net,net2)
        except Exception:
            pytest.fail(f"MLP failed losses(*nets) second pass")
            print(Exception)
    except Exception:
        pytest.fail(f"image_from_url failed")
        print(Exception)
    print("success")
except Exception:
    pytest.fail(f"import NeuralNetworks failed")
    print(Exception)