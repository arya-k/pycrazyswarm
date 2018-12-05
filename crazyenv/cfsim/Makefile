firmdir = ./crazyflie-firmware
modinc = $(firmdir)/src/modules/interface
modsrc = $(firmdir)/src/modules/src

swig:
	swig -python -I$(modinc) cffirmware.i
	gcc -std=c99 -shared -fPIC \
		-I$(modinc) -I/usr/include/python3.5 \
		$(modsrc)/planner.c $(modsrc)/pptraj.c \
		cffirmware_wrap.c \
		-lm -lpython3.5m \
		-o _cffirmware.so

clean:
	rm -f cffirmware.py cffirmware.pyc _cffirmware.so __init__.pyc __init__.py
