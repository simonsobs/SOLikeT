===============
GitHub Workflow
===============

If you are a membed of the `Simons Observatory GitHub organisation <https://github.com/simonsobs>`_ then contributions can be made directly to the `SOLikeT repo <https://github.com/simonsobs/soliket>`_.

If you are not a member of the Simons Observatory Collaboration with access to the GitHub org, you can still make contributions via forking this repo, with additional instructions below.

If you are planning on bringing a new Likelihood in to SOLikeT, please have a look at the `guidelines <https://github.com/simonsobs/SOLikeT/blob/master/guidelines.md>`_ which explain how this should be done.

Clone the SOLikeT Repository
============================

**You should only need to do this step once**

Cloning creates a local copy of the repository on your computer to work with. To clone your fork:

::

   git clone https://github.com/simonsobs/soliket.git

**If you are not an SO GitHub org member**, first *fork* the SOLikeT repository. A fork is your own remote copy of the repository on GitHub. To create a fork:

  1. Go to the `SOLikeT GitHub Repository <https://github.com/simonsobs/soliket>`_
  2. Click the **Fork** button (in the top-right-hand corner)
  3. Choose where to create the fork, typically your personal GitHub account

Next *clone* your fork. Cloning creates a local copy of the repository on your computer to work with. To clone your fork:

::

   git clone https://github.com/<your-account>/soliket.git


Finally add the ``simonsobs`` repository as a *remote*. This will allow you to fetch changes made to the codebase. To add the ``simonsobs`` remote:

::

  cd soliket
  git remote add simonsobs https://github.com/simonsobs/soliket.git


Create a branch for your new feature
====================================

Create a *branch* off the master branch. Working on unique branches for each new feature simplifies the development, review and merge processes by maintining logical separation. Branch names should have names prefixed with `dev-` and reflect the work going on within them. To create a feature branch:

::

  git checkout -b <your-branch-name> master


**If you are not an SO GitHub org member**, create a *branch* off the ``simonsobs`` master branch:

::

  git fetch simonsobs
  git checkout -b <your-branch-name> simonsobs/master


Hack away!
==========

Write the new code you would like to contribute, remembering to abide by the `Code Style`_, and *commit* it to the feature branch on your local repository. Ideally commit small units of work often with clear and descriptive commit messages describing the changes you made. To commit changes to a file:

::

  git add file_containing_your_contribution
  git commit -m 'Your clear and descriptive commit message'

*Push* the contributions in your feature branch to your remote fork on GitHub:

::

  git push origin <your-branch-name>


**Note:** The first time you *push* a feature branch you will probably need to use `--set-upstream origin` to link to your remote fork:

::

  git push --set-upstream origin <your-branch-name>


Open a Pull Request
===================

When you feel that work on your new feature is complete, you should create a *Pull Request*. This will propose your work to be merged into the main SOLikeT repository. If you would like feedback from others on active work in progress, even at an early stage, you can create a 'Draft' Pull Request at step 7. by choosing it from the drop-down menu. This will allow others to see and comment on your PR work in progress, which can be very helpful in getting it finished!

  1. Go to `SOLikeT Pull Requests <https://github.com/simonsobs/soliket/pulls>`_
  2. Click the green **New pull request** button
  3. Click **compare across forks**
  4. Confirm that the base fork is ``simonsobs/soliket`` and the base branch is ``master``
  5. If you are making a contribution from a fork, confirm the head fork is ``<your-account>/soliket`` and the compare branch is ``<your-branch-name>``
  6. Give your pull request a title and fill out the the template for the description
  7. Click the green **Create pull request** button


Status checks
=============

A series of automated checks will be run on your pull request, some of which will be required to pass before it can be merged into the main codebase:

  - ``Tests`` (Required) runs the `unit tests`
  - ``Code Style`` (Required) runs `flake8 <https://flake8.pycqa.org/en/latest/>`__ to check that your code conforms to the `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ style guidelines. Click "Details" to view any errors.
..
  _ in four predefined environments; `latest supported versions`, `oldest supported versions`, `macOS latest supported` and `Windows latest supported`. Click "Details" to view the output including any failures.

  - ``codecov`` reports the test coverage for your pull request; you should aim for `codecov/patch — 100.00%`. Click "Details" to view coverage data.
  - ``docs`` (Required) builds the `docstrings`_ on `readthedocs <https://readthedocs.org/>`_. Click "Details" to view the documentation or the failed build log.

Updating your branch
====================

As you work on your feature, new commits might be made to the ``simonsobs`` master branch. You will need to update your branch with these new commits before your pull request can be accepted. You can achieve this in a few different ways:

  - If your pull request has no conflicts, click **Update branch**
  - If your pull request has conflicts, click **Resolve conflicts**, manually resolve the conflicts and click **Mark as resolved**
  - *merge* the master branch into your dev branch from the command line:

    ::

        git fetch 
        git merge master

  - *rebase* your feature branch onto the ``simonsobs`` master branch from the command line:

    ::

        git fetch
        git rebase master

  - **If you are working on a fork** you will also need to specify that you updating from the ``simonsobs`` master branch:

    ::

        git fetch simonsobs
        git merge simonsobs/master

        git fetch simonsobs
        git rebase simonsobs/master


**Warning**: You should take care to take this step and pull other contributors work to your branch before attempting any rebase.

For more information about resolving conflicts see the GitHub guides:
  - `Resolving a merge conflict on GitHub <https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/resolving-a-merge-conflict-on-github>`_
  - `Resolving a merge conflict using the command line <https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/resolving-a-merge-conflict-using-the-command-line>`_
  - `About Git rebase <https://help.github.com/en/github/using-git/about-git-rebase>`_

More Information
================

More information regarding the usage of GitHub can be found in the `GitHub Guides <https://guides.github.com/>`_.
