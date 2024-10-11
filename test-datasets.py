from datasets import load_dataset

mlsum = load_dataset('mlsum', 'es')

print(mlsum['train'][1])