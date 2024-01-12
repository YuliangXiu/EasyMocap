import os

genders = {
        '00145': 'male', '00170': 'female', '00182': 'female', '00189': 'male', '00228': 'male',
        '00270': 'female', '03303': 'female', '03362': 'female', '03375': 'male', '03452': 'female',
        '03526': 'female', '03539': 'female', '03581': 'male', '03584': 'female', '03588': 'female',
        '03589': 'female', '03590': 'female', '03591': 'female', '03592': 'male', '03594': 'female',
        '03595': 'female', '03598': 'female', '03599': 'female', '03603': 'female', '03604': 'male',
        '03607': 'male', '03612': 'male', '03614': 'male', '03615': 'male', '03616': 'male',
        '03617': 'female', '03618': 'female', '03619': 'female', '03621': 'female', '03625':
        'female', '03626': 'female', '03627': 'male', '03628': 'male', '03629': 'male', '03630':
        'male', '03631': 'male', '03632': 'male', '03633': 'male'
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tgt_path', type=str, help="the path of data")
    parser.add_argument('--job_path', type=str, help="the path of job file")
    args = parser.parse_args()
    
    subjects = sorted(os.listdir(args.tgt_path))
    
    with open(args.job_path, 'w') as f:
        for subject in subjects:
            motions = sorted(os.listdir(os.path.join(args.tgt_path, subject)))
            for motion in motions:
                if os.path.isdir(os.path.join(args.tgt_path, subject, motion)):
                    f.write(f"{subject}/{motion} {genders[subject]} {subject}_{motion}\n")
            