from django.core.management.base import BaseCommand, CommandError
from django.contrib.auth import get_user_model
from expenses.models import Expense, Category
import pandas as pd
import random
import datetime


class Command(BaseCommand):
    help = 'Import sample expenses from dataset.csv into the Expense model for a given user.'

    def add_arguments(self, parser):
        parser.add_argument('--username', type=str, help='Username to assign imported expenses to', required=True)
        parser.add_argument('--count', type=int, help='Number of expenses to create', default=30)
        parser.add_argument('--start-days-ago', type=int, help='Earliest date (days ago) for random dates', default=60)

    def handle(self, *args, **options):
        username = options['username']
        count = options['count']
        start_days = options['start_days_ago']

        User = get_user_model()
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            raise CommandError(f"User with username '{username}' does not exist.")

        # Read dataset
        try:
            df = pd.read_csv('dataset.csv')
        except FileNotFoundError:
            raise CommandError("dataset.csv not found in project root.")

        if 'description' not in df.columns or 'category' not in df.columns:
            raise CommandError('dataset.csv must contain description and category columns')

        descriptions = df['description'].astype(str).tolist()
        categories = df['category'].astype(str).tolist()
        rows = list(zip(descriptions, categories))

        created = 0
        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=start_days)

        for i in range(count):
            desc, cat = random.choice(rows)
            # Normalize category name to titlecase for Category model
            cat_name = str(cat).strip().title()
            # Ensure a Category record exists (optional but helpful)
            try:
                Category.objects.get_or_create(name=cat_name)
            except Exception:
                # If migrations not applied or DB locked, continue using string
                pass

            # Generate a random amount based on category heuristics
            base_amount = 50
            cat_lower = str(cat).lower()
            if 'food' in cat_lower or 'grocery' in cat_lower:
                amount = round(random.uniform(20, 500), 2)
            elif 'transport' in cat_lower or 'uber' in cat_lower or 'bus' in cat_lower:
                amount = round(random.uniform(20, 300), 2)
            elif 'entertain' in cat_lower or 'concert' in cat_lower:
                amount = round(random.uniform(100, 2000), 2)
            elif 'rent' in cat_lower or 'mortage' in cat_lower or 'housing' in cat_lower:
                amount = round(random.uniform(5000, 30000), 2)
            elif 'health' in cat_lower or 'medical' in cat_lower:
                amount = round(random.uniform(100, 5000), 2)
            else:
                amount = round(random.uniform(20, 1500), 2)

            # Random date between start_date and today
            delta_days = (today - start_date).days
            rand_days = random.randint(0, max(0, delta_days))
            expense_date = start_date + datetime.timedelta(days=rand_days)

            Expense.objects.create(
                owner=user,
                amount=amount,
                date=expense_date,
                category=cat_name,
                description=str(desc),
            )
            created += 1

        self.stdout.write(self.style.SUCCESS(f'Successfully created {created} expenses for user {username}'))
