import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
import os
import time
import glob

# import torch
# import torch.nn as nn

# from torchtext.legacy import data
# from torchtext.legacy import datasets

from model import SNLIClassifier
from util import get_args, makedirs


args = get_args()
# 'torch.cuda.is_available' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)  # 'torch.cuda.set_device' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    device = torch.device('cuda:{}'.format(args.gpu))  # 'torch.device' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
# 'torch.backends.mps.is_available' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
elif torch.backends.mps.is_available():
    device = torch.device('mps')  # 'torch.device' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
else:
    device = torch.device('cpu')  # 'torch.device' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

inputs = data.Field(lower=args.lower, tokenize='spacy')  # 'torchtext.legacy.data.Field' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
answers = data.Field(sequential=False)  # 'torchtext.legacy.data.Field' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

train, dev, test = datasets.SNLI.splits(inputs, answers)  # 'torchtext.legacy.datasets.SNLI.splits' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

inputs.build_vocab(train, dev, test)
if args.word_vectors:
    if os.path.isfile(args.vector_cache):
        inputs.vocab.vectors = torch.load(args.vector_cache)  # 'torch.load' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    else:
        inputs.vocab.load_vectors(args.word_vectors)
        makedirs(os.path.dirname(args.vector_cache))
        torch.save(inputs.vocab.vectors, args.vector_cache)  # 'torch.save' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
answers.build_vocab(train)

train_iter, dev_iter, test_iter = data.BucketIterator.splits(
            (train, dev, test), batch_size=args.batch_size, device=device)  # 'torchtext.legacy.data.BucketIterator.splits' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

config = args
config.n_embed = len(inputs.vocab)
config.d_out = len(answers.vocab)
config.n_cells = config.n_layers

# double the number of cells for bidirectional networks
if config.birnn:
    config.n_cells *= 2

if args.resume_snapshot:
    model = torch.load(args.resume_snapshot, map_location=device)  # 'torch.load' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
else:
    model = SNLIClassifier(config)
    if args.word_vectors:
        model.embed.weight.data.copy_(inputs.vocab.vectors)
        model.to(device)

criterion = nn.CrossEntropyLoss()
opt = mint.optim.Adam(model.parameters(), lr=args.lr)

iterations = 0
start = time.time()
best_dev_acc = -1
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template =     ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
makedirs(args.save_path)
print(header)

for epoch in range(args.epochs):
    train_iter.init_epoch()
    n_correct, n_total = 0, 0
    for batch_idx, batch in enumerate(train_iter):

        # switch model to training mode, clear gradient accumulators
        model.train(); opt.zero_grad()

        iterations += 1

        # forward pass
        answer = model(batch)

        # calculate accuracy of predictions in the current batch
        n_correct += (mint.max(answer, 1)[1].view(batch.label.size()) == batch.label).sum().item()
        n_total += batch.batch_size
        train_acc = 100. * n_correct/n_total

        # calculate loss of the network output with respect to training labels
        loss = criterion(answer, batch.label)

        # backpropagate and update optimizer learning rate
        loss.backward(); opt.step()

        # checkpoint model periodically
        if iterations % args.save_every == 0:
            snapshot_prefix = os.path.join(args.save_path, 'snapshot')
            snapshot_path = snapshot_prefix + '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(train_acc, loss.item(), iterations)
            torch.save(model, snapshot_path)  # 'torch.save' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            for f in glob.glob(snapshot_prefix + '*'):
                if f != snapshot_path:
                    os.remove(f)

        # evaluate performance on validation set periodically
        if iterations % args.dev_every == 0:

            # switch model to evaluation mode
            model.eval(); dev_iter.init_epoch()

            # calculate accuracy on validation set
            n_dev_correct, dev_loss = 0, 0
            # 'torch.no_grad' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            with torch.no_grad():
                for dev_batch_idx, dev_batch in enumerate(dev_iter):
                     answer = model(dev_batch)
                     n_dev_correct += (mint.max(answer, 1)[1].view(dev_batch.label.size()) == dev_batch.label).sum().item()
                     dev_loss = criterion(answer, dev_batch.label)
            dev_acc = 100. * n_dev_correct / len(dev)

            print(dev_log_template.format(time.time()-start,
                epoch, iterations, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx) / len(train_iter), loss.item(), dev_loss.item(), train_acc, dev_acc))

            # update best validation set accuracy
            if dev_acc > best_dev_acc:

                # found a model with better validation set accuracy

                best_dev_acc = dev_acc
                snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
                snapshot_path = snapshot_prefix + '_devacc_{}_devloss_{}__iter_{}_model.pt'.format(dev_acc, dev_loss.item(), iterations)

                # save model, delete previous 'best_snapshot' files
                torch.save(model, snapshot_path)  # 'torch.save' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

        elif iterations % args.log_every == 0:

            # print progress message
            print(log_template.format(time.time()-start,
                epoch, iterations, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx) / len(train_iter), loss.item(), ' '*8, n_correct/n_total*100, ' '*12))
        if args.dry_run:
            break
