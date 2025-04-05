import os

def check_template_exists(template_name):
    template_path = os.path.join('templates', template_name)
    if os.path.isfile(template_path):
        print(f"{template_name} exists in the templates directory.")
    else:
        print(f"{template_name} does not exist in the templates directory.")

if __name__ == '__main__':
    check_template_exists('dashboard.html')
