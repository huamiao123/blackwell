import sys, os
src = sys.argv[1]
data = open(src, 'rb').read()
i = data.find(b'\x7fELF')
print(src, 'len', len(data), 'elf_at', i)
if i >= 0:
    out = '/root/fp4_bf16_study/raw/' + os.path.basename(src) + '.cubin'
    open(out, 'wb').write(data[i:])
    print('wrote', out, os.path.getsize(out))