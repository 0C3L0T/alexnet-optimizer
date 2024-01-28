
## Downloading Kernel source code
git clone https://github.com/khadas/linux.git -b khadas-vims-pie --depth 1
sed -i 's/^YYLTYPE yylloc;/extern YYLTYPE yylloc;/g' linux/scripts/dtc/dtc-lexer.lex.c

## Downloading tool-chains for cross-compilation
cd linux
toolchains_dir=/opt/toolchains_lab
wget --no-check-certificate https://releases.linaro.org/components/toolchain/binaries/6.3-2017.02/arm-linux-gnueabihf/gcc-linaro-6.3.1-2017.02-x86_64_arm-linux-gnueabihf.tar.xz
sudo mkdir ${toolchains_dir}
sudo tar xvJf gcc-linaro-6.3.1-2017.02-x86_64_arm-linux-gnueabihf.tar.xz -C ${toolchains_dir}
## install build-essential libraries (such as make)
## doesn't work on arch
#sudo apt-get update
#sudo apt install build-essential

## Setting parameters then make 
export CROSS_COMPILE=${toolchains_dir}/gcc-linaro-6.3.1-2017.02-x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf-
export ARCH=arm

make kvim_a32_defconfig
make prepare
make amlogic/kvim3.dtb

echo "Kernel source code directory:"
echo `pwd`
