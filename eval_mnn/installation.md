To install the C++ version, you can follow the instructions [here](https://qengineering.eu/install-mnn-on-raspberry-pi-4.html), they are pretty much plug and play. 
However, the installation process for their python binding is even more tedious compare to installing torch.

Here is the list of procedure that I collected and modified from the discussion of their [issues board](https://github.com/alibaba/MNN/issues/1051#issuecomment-780978045):

1. install dependency for compile
    ```
    sudo apt-get install cmake libprotobuf-dev protobuf-compiler
    ```
2. get the source code
    ```
    git clone https://github.com/alibaba/MNN
    ```
3. compile preparation
    ```
    cd /path/to/MNN
    ./schema/generate.sh
    mkdir pymnn_build
    ```
    Some of the error that appeared in previous didn't occur during my installation process. 

4. build dependency
    ```
    cd pymnn/pip_package
    python build_deps.py
    ```
5. build the wheel, but before, there's one last file needs to be modified
    ```python
    pymnn/pip_package/build_wheel.py
    ...
        if IS_LINUX:
            os.system('python setup.py bdist_wheel --plat-name=manylinux1_x86_64')
    ...
    ```
    to (for 64-bit Raspberry Pi OS)
    ```python
    pymnn/pip_package/build_wheel.py
    ...
        if IS_LINUX:
            os.system('python setup.py bdist_wheel --plat-name=manylinux2014_aarch64')
    ...
    ```
    Also, don't build `_tools`, comment out the following lines
    ```python
    pymnn/pip_package/setup.py
    ...
        tools = Extension("_tools",\
                        libraries=tools_libraries,\
                        sources=tools_sources,\
                        language='c++',\
                        extra_compile_args=tools_compile_args + extra_compile_args,\
                        include_dirs=tools_include_dirs,\
                        library_dirs=tools_library_dirs,\
                        extra_link_args=tools_extra_link_args +tools_link_args\
                            + [make_relative_rpath('lib')])
        extensions.append(tools)
    ...
    ```
    build wheel
    ```
    python build_wheel.py --version 2.7.1
    ```
6. install wheel
    ```
    pip install pymnn/pip_package/dist/MNN-2.7.1-cp39-cp39-manylinux2014_aarch64.whl
    ```
7. test installation
    ```python
    import MNN
    ```