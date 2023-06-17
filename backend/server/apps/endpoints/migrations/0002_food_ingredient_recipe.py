# Generated by Django 4.2 on 2023-05-21 13:50

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('endpoints', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Food',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('kcal_number', models.DecimalField(decimal_places=2, max_digits=6)),
            ],
        ),
        migrations.CreateModel(
            name='Ingredient',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('food', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='endpoints.food')),
                ('substitute', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='endpoints.ingredient')),
            ],
        ),
        migrations.CreateModel(
            name='Recipe',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('quantity', models.IntegerField(max_length=10)),
                ('ingredient1', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='endpoints.ingredient')),
            ],
        ),
    ]
